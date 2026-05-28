@tool
extends CompositorEffect
class_name SSAO


var rd: RenderingDevice

var hbao_shader: RID
var hbao_pipeline: RID

var blur_shader: RID
var blur_pipeline: RID

var blur_vertical_shader: RID
var blur_vertical_pipeline: RID

var compose_shader: RID
var compose_pipeline: RID

var nearest_sampler: RID
var linear_sampler: RID

var ssao_image: RID
var blur_image_1: RID
var blur_image_2: RID

var framebuffer_size : Vector2i = Vector2i(0, 0)

var mat_buffer : RID
var scene_buffer: RID

var settings : PackedFloat32Array
var settings_dirty : bool = false
var matrix_dirty : bool = false

var mutex: Mutex = Mutex.new()

@export_range(0.0,89.0,1.0) var Bias: float = 15.0 :
	set(New_Bias):
		Bias = New_Bias
		settings_dirty = true
@export_range(0.0,8.0,0.01) var Strength_Small_Scale: float =  1.0 :
	set(New_Strength_Small_Scale):
		Strength_Small_Scale = New_Strength_Small_Scale
		settings_dirty = true
@export_range(0.1,8.0,0.01) var Strength_Large_Scale: float =  1.0 :
	set(New_Strength_Large_Scale):
		Strength_Large_Scale = New_Strength_Large_Scale
		settings_dirty = true
@export_range(0.1,8.0,0.01) var Radius: float = 1.0 :
	set(New_Radius):
		Radius = New_Radius
		settings_dirty = true
@export_range(0.0,100.0,0.01) var Filter_Sharpness: float = 40.0 :
	set(New_Filter_Sharpness):
		Filter_Sharpness = New_Filter_Sharpness
		settings_dirty = true
@export_range(0.0,8.0,0.01) var Power_exponent: float = 2.0 :
	set(New_Power_exponent):
		Power_exponent = New_Power_exponent
		settings_dirty = true

func _init() -> void:
	effect_callback_type = EFFECT_CALLBACK_TYPE_POST_TRANSPARENT
	rd = RenderingServer.get_rendering_device()
	RenderingServer.call_on_render_thread(_initialize_compute)

	var sampler_state := RDSamplerState.new()
	sampler_state.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state.min_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	sampler_state.mag_filter = RenderingDevice.SAMPLER_FILTER_NEAREST
	nearest_sampler = RenderingServer.get_rendering_device().sampler_create(sampler_state)

	var sampler_state_linear := RDSamplerState.new()
	sampler_state_linear.repeat_u = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state_linear.repeat_v = RenderingDevice.SAMPLER_REPEAT_MODE_REPEAT
	sampler_state_linear.min_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	sampler_state_linear.mag_filter = RenderingDevice.SAMPLER_FILTER_LINEAR
	linear_sampler = RenderingServer.get_rendering_device().sampler_create(sampler_state_linear)

	var size: Vector2i = Vector2i(rd.screen_get_width(),rd.screen_get_height())

	_create_textures(size)
	_create_settings_buffer()



func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE:
			
		if hbao_shader.is_valid():
			rd.free_rid(hbao_shader)
			
		if blur_shader.is_valid():
			rd.free_rid(blur_shader)
			
		if blur_vertical_shader.is_valid():
			rd.free_rid(blur_vertical_shader)
			
		if compose_shader.is_valid():
			rd.free_rid(compose_shader)
			

func _clean_textures() -> void:
	if blur_image_1.is_valid():
		rd.free_rid(blur_image_1)
		blur_image_1 = RID()
		
	if blur_image_2.is_valid():
		rd.free_rid(blur_image_2)
		blur_image_2 = RID()
		
	if ssao_image.is_valid():
		rd.free_rid(ssao_image)
		ssao_image = RID()
	

func _create_settings_buffer():
	if scene_buffer.is_valid():
		rd.free_rid(scene_buffer)

	settings = PackedFloat32Array([
		Bias,
		Strength_Small_Scale,
		Radius,
		Filter_Sharpness,
		Power_exponent,
		0.0,
		0.0,
		Strength_Large_Scale
	])

	var db = PackedByteArray()

	db.append_array(settings.to_byte_array())
	scene_buffer =  rd.uniform_buffer_create(db.size(), db)

func _create_matrix_buffer(render_scene_data, view):
	if mat_buffer.is_valid():
		rd.free_rid(mat_buffer)

	var cam = render_scene_data.get_cam_projection()
	var view_proj = render_scene_data.get_view_projection(view)


	var cam_mat = [
		cam.x.x, cam.x.y, cam.x.z, cam.x.w,
		cam.y.x, cam.y.y, cam.y.z, cam.y.w,
		cam.z.x, cam.z.y, cam.z.z, cam.z.w,
		cam.w.x, cam.w.y, cam.w.z, cam.w.w,
	]

	var proj_mat = [
		view_proj.x.x, view_proj.x.y, view_proj.x.z, view_proj.x.w,
		view_proj.y.x, view_proj.y.y, view_proj.y.z, view_proj.y.w,
		view_proj.z.x, view_proj.z.y, view_proj.z.z, view_proj.z.w, 
		view_proj.w.x, view_proj.w.y, view_proj.w.z, view_proj.w.w,
	]

	var cma = PackedFloat32Array(cam_mat).to_byte_array()
	var vpa = PackedFloat32Array(proj_mat).to_byte_array()


	var pb = PackedByteArray()
	pb.append_array(cma)
	pb.append_array(vpa)

	mat_buffer =  rd.uniform_buffer_create(128, pb)

func _create_textures(size: Vector2i) -> void:
	var txt = RDTextureFormat.new()
	txt.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
	txt.width = size.x
	txt.height = size.y
	txt.depth = 1
	txt.mipmaps = 1
	txt.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT + RenderingDevice.TEXTURE_USAGE_COLOR_ATTACHMENT_BIT + RenderingDevice.TEXTURE_USAGE_STORAGE_BIT + RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT + RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
	ssao_image = rd.texture_create(txt, RDTextureView.new())
	blur_image_1 = rd.texture_create(txt, RDTextureView.new())
	blur_image_2 = rd.texture_create(txt, RDTextureView.new())
	
#region Code in this region runs on the rendering thread.
# Compile our shader at initialization.
func _initialize_compute() -> void:
	rd = RenderingServer.get_rendering_device()
	if not rd:
		return
	
		
	var color_attachment_format : RDAttachmentFormat = RDAttachmentFormat.new()
	var no_blend_attachment := RDPipelineColorBlendStateAttachment.new()
	color_attachment_format.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
	color_attachment_format.usage_flags = RenderingDevice.TEXTURE_USAGE_COLOR_ATTACHMENT_BIT
	var color_framebuffer_format = rd.framebuffer_format_create([color_attachment_format])
	var no_blend := RDPipelineColorBlendState.new()
	no_blend.attachments = [no_blend_attachment]
	var stencil_state := RDPipelineDepthStencilState.new()
	var multisample_state := RDPipelineMultisampleState.new()
	var pipeline_state := RDPipelineRasterizationState.new()
	
	# Compile our shader.
	var shader_file_ssao := load("res://addons/Godot_HBAO/SSAO.glsl")
	var shader_spirv_ssao: RDShaderSPIRV = shader_file_ssao.get_spirv()

	hbao_shader = rd.shader_create_from_spirv(shader_spirv_ssao)
	if hbao_shader.is_valid():
		hbao_pipeline = rd.render_pipeline_create(hbao_shader,color_framebuffer_format,-1, RenderingDevice.RENDER_PRIMITIVE_TRIANGLES, pipeline_state,
			multisample_state, stencil_state,
			no_blend)

	var shader_file_blur_horizontal := load("res://addons/Godot_HBAO/blur_horizontal.glsl")
	var shader_spirv_blur_horizontal: RDShaderSPIRV = shader_file_blur_horizontal.get_spirv()

	blur_shader = rd.shader_create_from_spirv(shader_spirv_blur_horizontal)
	if blur_shader.is_valid():
		blur_pipeline = rd.render_pipeline_create(blur_shader,color_framebuffer_format,-1, RenderingDevice.RENDER_PRIMITIVE_TRIANGLES, pipeline_state,
			multisample_state, stencil_state,
			no_blend)
			
	var shader_file_blur_vertical := load("res://addons/Godot_HBAO/blur_vertical.glsl")
	var shader_spirv_blur_vertical: RDShaderSPIRV = shader_file_blur_vertical.get_spirv()

	blur_vertical_shader = rd.shader_create_from_spirv(shader_spirv_blur_vertical)
	if blur_vertical_shader.is_valid():
		blur_vertical_pipeline = rd.render_pipeline_create(blur_vertical_shader,color_framebuffer_format,-1, RenderingDevice.RENDER_PRIMITIVE_TRIANGLES, pipeline_state,
			multisample_state, stencil_state,
			no_blend)
			
	var shader_file_compose:= load("res://addons/Godot_HBAO/compose.glsl")
	var shader_spirv_compose: RDShaderSPIRV = shader_file_compose.get_spirv()

	compose_shader = rd.shader_create_from_spirv(shader_spirv_compose)

# Called by the rendering thread every frame.
func _render_callback(p_effect_callback_type: EffectCallbackType, p_render_data: RenderData) -> void:
	if rd and p_effect_callback_type == EFFECT_CALLBACK_TYPE_POST_TRANSPARENT:
		# Get our render scene buffers object, this gives us access to our render buffers.
		# Note that implementation differs per renderer hence the need for the cast.
		var render_scene_buffers := p_render_data.get_render_scene_buffers()
		if render_scene_buffers:
			# Get our render size, this is the 3D render resolution!
			var size: Vector2i = render_scene_buffers.get_internal_size()
			if size.x == 0 and size.y == 0:
				return
				
			var half_size := Vector2i((size.x) / 2.0, (size.y) / 2.0)
				
			mutex.lock()
			
			if size != framebuffer_size:
				framebuffer_size = size
				_clean_textures()
				_create_textures(half_size)
				matrix_dirty = true
			
			if settings_dirty == true:
				_create_settings_buffer()
				settings_dirty = false
			
			mutex.unlock()
			
			# Create push constant.
			# Must be aligned to 16 bytes and be in the same order as defined in the shader.
			var push_constant := PackedFloat32Array([
				half_size.x,
				half_size.y,
				0.0,
				0.0
			])
			
			# Loop through views just in case we're doing stereo rendering. No extra cost if this is mono.
			var view_count: int = render_scene_buffers.get_view_count()
			var render_scene_data = p_render_data.get_render_scene_data()
			for view in view_count:
				# Get the RID for our color image, we will be reading from and writing to it.
				var input_image: RID = render_scene_buffers.get_color_layer(view)
				var depth_image: RID = render_scene_buffers.get_depth_layer(view)
			
				var depth_uniform := RDUniform.new()
				depth_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
				depth_uniform.binding = 0
				depth_uniform.add_id(nearest_sampler)
				depth_uniform.add_id(depth_image)
				
				var depth_set = UniformSetCacheRD.get_cache(hbao_shader, 0, [depth_uniform])
				
				
				if !mat_buffer.is_valid():
					_create_matrix_buffer(render_scene_data, view)
					
				if matrix_dirty == true:
					_create_matrix_buffer(render_scene_data, view)
					matrix_dirty = false
				
				var matrices_uniform : RDUniform = RDUniform.new()
				matrices_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER
				matrices_uniform.binding = 0
				matrices_uniform.add_id(mat_buffer)
				var matrices_set = UniformSetCacheRD.get_cache(hbao_shader, 2, [matrices_uniform])
				
				var scene_uniform : RDUniform = RDUniform.new()
				scene_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER
				scene_uniform.binding = 0
				scene_uniform.add_id(scene_buffer)
				var scene_set = UniformSetCacheRD.get_cache(hbao_shader, 3, [scene_uniform])
				
				var ssao_framebuffer = FramebufferCacheRD.get_cache_multipass([ssao_image],[],1)
				var ssao_draw_list := rd.draw_list_begin(ssao_framebuffer,RenderingDevice.DRAW_IGNORE_ALL,);
				rd.draw_list_bind_render_pipeline(ssao_draw_list, hbao_pipeline)
				rd.draw_list_bind_uniform_set(ssao_draw_list, depth_set, 0)
				rd.draw_list_bind_uniform_set(ssao_draw_list, matrices_set, 2)
				rd.draw_list_bind_uniform_set(ssao_draw_list, scene_set, 3)
				rd.draw_list_set_push_constant(ssao_draw_list, push_constant.to_byte_array(), push_constant.size() * 4)
				rd.draw_list_draw(ssao_draw_list, false, 1, 3)
				rd.draw_list_end()
				
				scene_set = UniformSetCacheRD.get_cache(blur_shader, 3, [scene_uniform])
				
				var ssao_uniform := RDUniform.new()
				ssao_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
				ssao_uniform.binding = 0
				ssao_uniform.add_id(nearest_sampler)
				ssao_uniform.add_id(ssao_image)
				var ssao_set = UniformSetCacheRD.get_cache(blur_shader, 0, [ssao_uniform])
				
				var blur_1_framebuffer = FramebufferCacheRD.get_cache_multipass([blur_image_1],[],1)
				var blur_1_draw_list := rd.draw_list_begin(blur_1_framebuffer,RenderingDevice.DRAW_IGNORE_ALL,);
				rd.draw_list_bind_render_pipeline(blur_1_draw_list, blur_pipeline)
				rd.draw_list_bind_uniform_set(blur_1_draw_list, ssao_set, 0)
				rd.draw_list_bind_uniform_set(blur_1_draw_list, matrices_set, 2)
				rd.draw_list_bind_uniform_set(blur_1_draw_list, scene_set, 3)
				rd.draw_list_set_push_constant(blur_1_draw_list, push_constant.to_byte_array(), push_constant.size() * 4)
				rd.draw_list_draw(blur_1_draw_list, false, 1, 3)
				rd.draw_list_end()
				
				var blur_uniform_1 := RDUniform.new()
				blur_uniform_1.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
				blur_uniform_1.binding = 0
				blur_uniform_1.add_id(nearest_sampler)
				blur_uniform_1.add_id(blur_image_1)
				var blur_1_set = UniformSetCacheRD.get_cache(blur_shader, 0, [blur_uniform_1])
				
				var blur_2_framebuffer = FramebufferCacheRD.get_cache_multipass([blur_image_2],[],1)
				var blur_2_draw_list := rd.draw_list_begin(blur_2_framebuffer,RenderingDevice.DRAW_IGNORE_ALL,);
				rd.draw_list_bind_render_pipeline(blur_2_draw_list, blur_vertical_pipeline)
				rd.draw_list_bind_uniform_set(blur_2_draw_list, blur_1_set, 0)
				rd.draw_list_bind_uniform_set(blur_2_draw_list, matrices_set, 2)
				rd.draw_list_bind_uniform_set(blur_2_draw_list, scene_set, 3)
				rd.draw_list_set_push_constant(blur_2_draw_list, push_constant.to_byte_array(), push_constant.size() * 4)
				rd.draw_list_draw(blur_2_draw_list, false, 1, 3)
				rd.draw_list_end()
				
				var fb: RID = FramebufferCacheRD.get_cache_multipass([input_image], [], 1)
				var fb_format := rd.framebuffer_get_format(fb)
				
				var combine_uniform := RDUniform.new()
				combine_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
				combine_uniform.binding = 0
				combine_uniform.add_id(linear_sampler)
				combine_uniform.add_id(blur_image_2)
				
				var combine_uniform_set = UniformSetCacheRD.get_cache(compose_shader, 0, [combine_uniform])
				
				
				if not compose_pipeline.is_valid():
					var color_attachment_format : RDAttachmentFormat = RDAttachmentFormat.new()
					var rasterization_state = RDPipelineRasterizationState.new()
					var multisample_state = RDPipelineMultisampleState.new()
					var depth_stencil_state = RDPipelineDepthStencilState.new()
					var blend_state = RDPipelineColorBlendState.new()
					
					var blend_attachment = RDPipelineColorBlendStateAttachment.new()
					blend_attachment.enable_blend = true
					blend_attachment.alpha_blend_op = RenderingDevice.BLEND_OP_ADD
					blend_attachment.color_blend_op = RenderingDevice.BLEND_OP_ADD
					blend_attachment.src_color_blend_factor = RenderingDevice.BLEND_FACTOR_DST_COLOR
					blend_attachment.dst_color_blend_factor = RenderingDevice.BLEND_FACTOR_ZERO
					blend_attachment.src_alpha_blend_factor = RenderingDevice.BLEND_FACTOR_DST_ALPHA
					blend_attachment.dst_alpha_blend_factor = RenderingDevice.BLEND_FACTOR_ZERO
					
					blend_state.attachments = [blend_attachment]
					
					compose_pipeline = rd.render_pipeline_create(compose_shader, fb_format
					, RenderingDevice.INVALID_FORMAT_ID, RenderingDevice.RENDER_PRIMITIVE_TRIANGLES
					, rasterization_state, multisample_state, depth_stencil_state, blend_state);
					
				var clear_colors := PackedColorArray()
				var combine_draw_list := rd.draw_list_begin(fb, 0, clear_colors);
				rd.draw_list_bind_render_pipeline(combine_draw_list, compose_pipeline)
				rd.draw_list_bind_uniform_set(combine_draw_list, combine_uniform_set, 0)
				rd.draw_list_draw(combine_draw_list, false, 1, 3)
				
				rd.draw_list_end()
				
				rd.draw_command_end_label()

#endregion

func dispatch_stage(stage : RID, pipeline : RID, uniforms : Array[RDUniform], scene : Array[RDUniform], matricies : Array[RDUniform], push_constants : PackedByteArray, groups : Vector3):

	var matrices_uniform_set;
	var scene_set;

	var tex_uniform_set = UniformSetCacheRD.get_cache(stage, 0, uniforms)
	if matricies != null:
		matrices_uniform_set = UniformSetCacheRD.get_cache(stage, 2, matricies)
	if scene != null:
		scene_set = UniformSetCacheRD.get_cache(stage, 3, scene)
	
	var compute_list = rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(compute_list, pipeline)
	rd.compute_list_bind_uniform_set(compute_list, tex_uniform_set, 0)
	if matrices_uniform_set != null:
		rd.compute_list_bind_uniform_set(compute_list, matrices_uniform_set, 2)
	if scene_set != null:
		rd.compute_list_bind_uniform_set(compute_list, scene_set, 3)

	if !push_constants.is_empty():
		rd.compute_list_set_push_constant(compute_list, push_constants, push_constants.size())

	rd.compute_list_dispatch(compute_list, groups.x, groups.y, groups.z)

	rd.compute_list_end()

	rd.draw_command_end_label()

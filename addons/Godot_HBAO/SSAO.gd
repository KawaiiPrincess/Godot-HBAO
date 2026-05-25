@tool
extends CompositorEffect
class_name SSAO


var rd: RenderingDevice

var blit_shader: RID
var blit_pipeline: RID

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

var blit_image: RID
var ssao_image: RID
var blur_image_1: RID
var blur_image_2: RID
var noise_image: RID

var framebuffer_size : Vector2i = Vector2i(0, 0)

var mat_buffer : RID
var scene_buffer: RID

var settings : PackedFloat32Array
var settings_dirty : bool = false
var matrix_dirty : bool = false

var mutex: Mutex = Mutex.new()

@export_range(0.0,89.0,1.0) var Bias: float = 30.0 :
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
@export_range(1.0,100.0,0.01) var Filter_Sharpness: float = 5.0 :
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
		if blit_shader.is_valid():
			rd.free_rid(blit_shader)
			
		if hbao_shader.is_valid():
			rd.free_rid(hbao_shader)
			
		if blur_shader.is_valid():
			rd.free_rid(blur_shader)
			
		if blur_vertical_shader.is_valid():
			rd.free_rid(blur_vertical_shader)
			
		if compose_shader.is_valid():
			rd.free_rid(compose_shader)

func _clean_textures() -> void:
	if blit_image.is_valid():
		rd.free_rid(blit_image)
		blit_image = RID()
		
	if blur_image_1.is_valid():
		rd.free_rid(blur_image_1)
		blur_image_1 = RID()
		
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
		view_proj.z.x, view_proj.z.y, view_proj.z.z, view_proj.z.w, 			view_proj.w.x, view_proj.w.y, view_proj.w.z, view_proj.w.w,
	]

	var cma = PackedFloat32Array(cam_mat).to_byte_array()
	var vpa = PackedFloat32Array(proj_mat).to_byte_array()


	var pb = PackedByteArray()
	pb.append_array(cma)
	pb.append_array(vpa)

	mat_buffer =  rd.uniform_buffer_create(128, pb)

func _create_textures(size: Vector2i) -> void:
	var txt = RDTextureFormat.new()
	txt.format = RenderingDevice.DATA_FORMAT_R16G16B16A16_UNORM
	txt.width = size.x
	txt.height = size.y
	txt.depth = 1
	txt.mipmaps = 1
	txt.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT + RenderingDevice.TEXTURE_USAGE_COLOR_ATTACHMENT_BIT + RenderingDevice.TEXTURE_USAGE_STORAGE_BIT + RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT + RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
	
	blit_image = rd.texture_create(txt, RDTextureView.new())
	blur_image_1 = rd.texture_create(txt, RDTextureView.new())
	blur_image_2 = rd.texture_create(txt, RDTextureView.new())

	txt = RDTextureFormat.new()
	txt.format = RenderingDevice.DATA_FORMAT_R16G16B16A16_UNORM
	txt.width = size.x * 0.5
	txt.height = size.y * 0.5
	txt.depth = 1
	txt.mipmaps = 1
	txt.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT + RenderingDevice.TEXTURE_USAGE_COLOR_ATTACHMENT_BIT + RenderingDevice.TEXTURE_USAGE_STORAGE_BIT + RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT + RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
	ssao_image = rd.texture_create(txt, RDTextureView.new())
	
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

	var noise_tex = preload("res://addons/Godot_HBAO/HDR_RGB_0.png")
	noise_image = RenderingServer.texture_get_rd_texture(noise_tex.get_rid())
	
	# Compile our shader.
	var shader_file_blit := load("res://addons/Godot_HBAO/blit.glsl")
	var shader_spirv_blit: RDShaderSPIRV = shader_file_blit.get_spirv()

	blit_shader = rd.shader_create_from_spirv(shader_spirv_blit)
	if blit_shader.is_valid():
		blit_pipeline = rd.render_pipeline_create(blit_shader,color_framebuffer_format,-1, RenderingDevice.RENDER_PRIMITIVE_TRIANGLES, RDPipelineRasterizationState.new(),
			RDPipelineMultisampleState.new(), stencil_state,
			no_blend)
	
	var shader_file_ssao := load("res://addons/Godot_HBAO/SSAO.glsl")
	var shader_spirv_ssao: RDShaderSPIRV = shader_file_ssao.get_spirv()

	hbao_shader = rd.shader_create_from_spirv(shader_spirv_ssao)
	if hbao_shader.is_valid():
		hbao_pipeline = rd.render_pipeline_create(hbao_shader,color_framebuffer_format,-1, RenderingDevice.RENDER_PRIMITIVE_TRIANGLES, RDPipelineRasterizationState.new(),
			RDPipelineMultisampleState.new(), stencil_state,
			no_blend)

	var shader_file_blur_horizontal := load("res://addons/Godot_HBAO/blur_horizontal.glsl")
	var shader_spirv_blur_horizontal: RDShaderSPIRV = shader_file_blur_horizontal.get_spirv()

	blur_shader = rd.shader_create_from_spirv(shader_spirv_blur_horizontal)
	if blur_shader.is_valid():
		blur_pipeline = rd.render_pipeline_create(blur_shader,color_framebuffer_format,-1, RenderingDevice.RENDER_PRIMITIVE_TRIANGLES, RDPipelineRasterizationState.new(),
			RDPipelineMultisampleState.new(), stencil_state,
			no_blend)
			
	var shader_file_blur_vertical := load("res://addons/Godot_HBAO/blur_vertical.glsl")
	var shader_spirv_blur_vertical: RDShaderSPIRV = shader_file_blur_vertical.get_spirv()

	blur_vertical_shader = rd.shader_create_from_spirv(shader_spirv_blur_vertical)
	if blur_vertical_shader.is_valid():
		blur_vertical_pipeline = rd.render_pipeline_create(blur_vertical_shader,color_framebuffer_format,-1, RenderingDevice.RENDER_PRIMITIVE_TRIANGLES, RDPipelineRasterizationState.new(),
			RDPipelineMultisampleState.new(), stencil_state,
			no_blend)
			
	var shader_file_compose := load("res://addons/Godot_HBAO/compose.glsl")
	var shader_spirv_compose: RDShaderSPIRV = shader_file_compose.get_spirv()

	compose_shader = rd.shader_create_from_spirv(shader_spirv_compose)
	if compose_shader.is_valid():
		compose_pipeline = rd.render_pipeline_create(compose_shader,color_framebuffer_format,-1, RenderingDevice.RENDER_PRIMITIVE_TRIANGLES, RDPipelineRasterizationState.new(),
			RDPipelineMultisampleState.new(), stencil_state,
			no_blend)

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
				
			mutex.lock()
			
			if size != framebuffer_size:
				framebuffer_size = size
				_clean_textures()
				_create_textures(size)
				matrix_dirty = true
			
			if settings_dirty == true:
				_create_settings_buffer()
				settings_dirty = false
			
			mutex.unlock()
			
			# Create push constant.
			# Must be aligned to 16 bytes and be in the same order as defined in the shader.
			var push_constant := PackedFloat32Array([
				size.x,
				size.y,
				0.0,
				0.0
			])
			
			# Loop through views just in case we're doing stereo rendering. No extra cost if this is mono.
			var view_count: int = render_scene_buffers.get_view_count()
			var render_scene_data = p_render_data.get_render_scene_data()
			for view in view_count:
			
				if !mat_buffer.is_valid():
					_create_matrix_buffer(render_scene_data, view)
					
				if matrix_dirty == true:
					_create_matrix_buffer(render_scene_data, view)
					matrix_dirty = false
				
				var matrices_uniform : RDUniform = RDUniform.new()
				matrices_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER
				matrices_uniform.binding = 0
				matrices_uniform.add_id(mat_buffer)
				
				var scene_uniform : RDUniform = RDUniform.new()
				scene_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER
				scene_uniform.binding = 0
				scene_uniform.add_id(scene_buffer)
				
				# Get the RID for our color image, we will be reading from and writing to it.
				var input_image: RID = render_scene_buffers.get_color_layer(view)
				var depth_image: RID = render_scene_buffers.get_depth_layer(view)
				var ourput_image: RID = render_scene_buffers.get_color_layer(view)
				
				# Create a uniform set, this will be cached, the cache will be cleared if our viewports configuration is changed.
				var color_uniform := RDUniform.new()
				color_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
				color_uniform.binding = 1
				color_uniform.add_id(linear_sampler)
				color_uniform.add_id(input_image)
				
				var depth_uniform := RDUniform.new()
				depth_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
				depth_uniform.binding = 1
				depth_uniform.add_id(nearest_sampler)
				depth_uniform.add_id(depth_image)
				
				var blit_uniform := RDUniform.new()
				blit_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
				blit_uniform.binding = 1
				blit_uniform.add_id(linear_sampler)
				blit_uniform.add_id(blit_image)
				
				var ssao_uniform := RDUniform.new()
				ssao_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
				ssao_uniform.binding = 2
				ssao_uniform.add_id(linear_sampler)
				ssao_uniform.add_id(ssao_image)
				
				var blur_uniform_1 := RDUniform.new()
				blur_uniform_1.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
				blur_uniform_1.binding = 2
				blur_uniform_1.add_id(linear_sampler)
				blur_uniform_1.add_id(blur_image_1)
				
				var blur_uniform_2 := RDUniform.new()
				blur_uniform_2.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
				blur_uniform_2.binding = 2
				blur_uniform_2.add_id(linear_sampler)
				blur_uniform_2.add_id(blur_image_2)
				
				var noise_uniform := RDUniform.new()
				noise_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
				noise_uniform.binding = 3
				noise_uniform.add_id(nearest_sampler)
				noise_uniform.add_id(noise_image)
				
				dispatch_stage(
					blit_shader,
					blit_pipeline,
					[color_uniform],
					[scene_uniform],
					[matrices_uniform],
					push_constant.to_byte_array(),
					blit_image
				)
				
				dispatch_stage(
					hbao_shader,
					hbao_pipeline,
					[depth_uniform, noise_uniform],
					[scene_uniform],
					[matrices_uniform],
					push_constant.to_byte_array(),
					ssao_image
				)
				
				dispatch_stage(
					blur_shader,
					blur_pipeline,
					[depth_uniform, ssao_uniform],
					[scene_uniform],
					[matrices_uniform],
					push_constant.to_byte_array(),
					blur_image_1
				)
				
				dispatch_stage(
					blur_vertical_shader,
					blur_vertical_pipeline,
					[depth_uniform, blur_uniform_1],
					[scene_uniform],
					[matrices_uniform],
					push_constant.to_byte_array(),
					blur_image_2
				)
				
				dispatch_stage(
					compose_shader,
					compose_pipeline,
					[blit_uniform, blur_uniform_2],
					[scene_uniform],
					[matrices_uniform],
					push_constant.to_byte_array(),
					ourput_image
				)

#endregion

func dispatch_stage(stage : RID, pipeline : RID, uniforms : Array[RDUniform], scene : Array[RDUniform], matricies : Array[RDUniform], push_constants : PackedByteArray, output):

	var matrices_uniform_set;
	var scene_set;

	var tex_uniform_set = UniformSetCacheRD.get_cache(stage, 0, uniforms)
	if matricies != null:
		matrices_uniform_set = UniformSetCacheRD.get_cache(stage, 2, matricies)
	if scene != null:
		scene_set = UniformSetCacheRD.get_cache(stage, 3, scene)
	
	var copy_framebuffer = FramebufferCacheRD.get_cache_multipass([output], [], 1)
	var draw_list = rd.draw_list_begin(copy_framebuffer,RenderingDevice.DRAW_IGNORE_ALL,)
	rd.draw_list_bind_render_pipeline(draw_list, pipeline)
	rd.draw_list_bind_uniform_set(draw_list, tex_uniform_set, 0)
	if matrices_uniform_set != null:
		rd.draw_list_bind_uniform_set(draw_list, matrices_uniform_set, 2)
	if scene_set != null:
		rd.draw_list_bind_uniform_set(draw_list, scene_set, 3)

	if !push_constants.is_empty():
		rd.draw_list_set_push_constant(draw_list, push_constants, push_constants.size())

	rd.draw_list_draw(draw_list, false, 1, 3)

	rd.draw_list_end()

	rd.draw_command_end_label()

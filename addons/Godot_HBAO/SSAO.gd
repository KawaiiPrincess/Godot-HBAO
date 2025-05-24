@tool
extends CompositorEffect
class_name SSAO


var rd: RenderingDevice

var hbao_shader: RID 
var hbao_pipeline: RID

var blur_shader: RID 
var blur_pipeline: RID

var nearest_sampler: RID
var linear_sampler: RID

var blur_image: RID
var noise_image: RID

var framebuffer_size : Vector2i = Vector2i(0, 0)
var scene_buffer: RID
var buffer_dirty : bool = true

var mutex: Mutex = Mutex.new()

@export_range(0.0,89.0,1.0) var Bias: float = 30.0
@export_range(0.0,8.0,0.01) var Strength_Small_Scale: float =  1.0
@export_range(0.1,8.0,0.01) var Strength_Large_Scale: float =  1.0
@export_range(0.1,8.0,0.01) var Radius: float = 1.0
@export_range(1.0,100.0,0.01) var Filter_Sharpness: float = 5.0
@export_range(0.0,8.0,0.01) var Power_exponent: float = 2.0

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
	


# System notifications, we want to react on the notification that
# alerts us we are about to be destroyed.
func _notification(what: int) -> void:
	if what == NOTIFICATION_PREDELETE:
		if hbao_shader.is_valid():
			rd.free_rid(hbao_shader)
		if blur_shader.is_valid():
			rd.free_rid(blur_shader)


func _clean_textures() -> void:
	# Associated framebuffers are dependent on these textures
	# they're freed with them
	if blur_image.is_valid():
		rd.free_rid(blur_image)
		blur_image = RID()
		
func _create_textures(size: Vector2i) -> void:
	var txt = RDTextureFormat.new()
	txt.format = RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM
	txt.width = size.x
	txt.height = size.y
	txt.depth = 1
	txt.mipmaps = 1
	txt.usage_bits = RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT + RenderingDevice.TEXTURE_USAGE_COLOR_ATTACHMENT_BIT + RenderingDevice.TEXTURE_USAGE_STORAGE_BIT + RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT + RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
	blur_image = rd.texture_create(txt, RDTextureView.new())

#region Code in this region runs on the rendering thread.
# Compile our shader at initialization.
func _initialize_compute() -> void:
	rd = RenderingServer.get_rendering_device()
	if not rd:
		return

	var noise_tex = preload("res://addons/Godot_HBAO/HDR_RGB_0.png")
	noise_image = RenderingServer.texture_get_rd_texture(noise_tex.get_rid())
	# Compile our shader.
	var shader_file := load("res://addons/Godot_HBAO/SSAO.glsl")
	var shader_spirv: RDShaderSPIRV = shader_file.get_spirv()

	hbao_shader = rd.shader_create_from_spirv(shader_spirv)
	if hbao_shader.is_valid():
		hbao_pipeline = rd.compute_pipeline_create(hbao_shader)
	
	var shader_file2 := load("res://addons/Godot_HBAO/blur.glsl")
	var shader_spirv2: RDShaderSPIRV = shader_file2.get_spirv()
	
	blur_shader = rd.shader_create_from_spirv(shader_spirv2)
	if blur_shader.is_valid():
		blur_pipeline = rd.compute_pipeline_create(blur_shader)

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
			
			
			@warning_ignore("integer_division")
			var x_groups := (size.x - 1) / 16 + 1
			@warning_ignore("integer_division")
			var y_groups := (size.y - 1) / 16 + 1
			var z_groups := 1
			
			mutex.lock()
			if size != framebuffer_size:
				framebuffer_size = size
				_clean_textures()
				_create_textures(size)
			mutex.unlock()
			
			# We can use a compute shader here.


			# Create push constant.
			# Must be aligned to 16 bytes and be in the same order as defined in the shader.
			var push_constant := PackedFloat32Array([
				size.x,
				size.y,
				0.0,
				0.0
			])
			
			var settings := PackedFloat32Array([
				Bias,
				Strength_Small_Scale,
				Radius,
				Filter_Sharpness,
				Power_exponent,
				0.0,
				0.0,
				Strength_Large_Scale
			])
				
			# Loop through views just in case we're doing stereo rendering. No extra cost if this is mono.
			var view_count: int = render_scene_buffers.get_view_count()
			var render_scene_data = p_render_data.get_render_scene_data()
			for view in view_count:
				
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
				
				var mat_buffer : RID =  rd.uniform_buffer_create(128, pb)
				
				var matrices_uniform : RDUniform = RDUniform.new()
				matrices_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER
				matrices_uniform.binding = 0
				matrices_uniform.add_id(mat_buffer)
				#var matrices_uniform_set: RID = UniformSetCacheRD.get_cache(hbao_shader, 2, [  matrices_uniform ])
				

				var db = PackedByteArray()
				
				db.append_array(settings.to_byte_array())
				scene_buffer =  rd.uniform_buffer_create(db.size(), db)
				var scene_uniform : RDUniform = RDUniform.new()
				scene_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_UNIFORM_BUFFER
				scene_uniform.binding = 0
				scene_uniform.add_id(scene_buffer)
				#var scene_set: RID = rd.uniform_set_create( [  scene_uniform ], hbao_shader, 3)
				
				# Get the RID for our color image, we will be reading from and writing to it.
				var input_image: RID = render_scene_buffers.get_color_layer(view)
				var depth_image: RID = render_scene_buffers.get_depth_layer(view)
								

				# Create a uniform set, this will be cached, the cache will be cleared if our viewports configuration is changed.
				var color_uniform := RDUniform.new()
				color_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
				color_uniform.binding = 0
				color_uniform.add_id(input_image)
				
				var depth_uniform := RDUniform.new()
				depth_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
				depth_uniform.binding = 1
				depth_uniform.add_id(nearest_sampler)
				depth_uniform.add_id(depth_image)
				
				var blur_uniform := RDUniform.new()
				blur_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
				blur_uniform.binding = 2
				blur_uniform.add_id(blur_image)
				
				var noise_uniform := RDUniform.new()
				noise_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_SAMPLER_WITH_TEXTURE
				noise_uniform.binding = 3
				noise_uniform.add_id(linear_sampler)
				noise_uniform.add_id(noise_image)
				#var uniform_set := UniformSetCacheRD.get_cache(hbao_shader, 0, [uniform, depth_uniform,blur_uniform, noise_uniform])
				# Run our compute shader.
				
				dispatch_stage(
					hbao_shader,
					hbao_pipeline,
					[depth_uniform, blur_uniform, noise_uniform],
					[scene_uniform],
					[matrices_uniform],
					push_constant.to_byte_array(),
					Vector3(x_groups, y_groups, z_groups)
				)
				
				blur_uniform = RDUniform.new()
				blur_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
				blur_uniform.binding = 2
				blur_uniform.add_id(blur_image)
				
				dispatch_stage(
					blur_shader,
					blur_pipeline,
					[color_uniform, depth_uniform, blur_uniform],
					[scene_uniform],
					[matrices_uniform],
					push_constant.to_byte_array(),
					Vector3(x_groups, y_groups, z_groups)
				)
				

#endregion

func dispatch_stage(stage : RID, pipeline : RID, uniforms : Array[RDUniform], scene : Array[RDUniform], matricies : Array[RDUniform], push_constants : PackedByteArray, dispatch_size : Vector3i):
	
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
		
	rd.compute_list_dispatch(compute_list, dispatch_size.x, dispatch_size.y, dispatch_size.z)
	
	rd.compute_list_end()
	
	rd.draw_command_end_label()

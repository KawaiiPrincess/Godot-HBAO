# Godot-HBAO
Custom SSAO effect for Godot, Implemented via Compositor Effect.

![Image](https://github.com/KawaiiPrincess/Godot-HBAO/blob/main/HBAO.png)

## Why?
Godots built in SSAO effect was too slow for my project so I decided to create a custom SSAO effect. It costs less than 1 ms per frame on my GPU (NVIDIA GTX 1650). Most of the performance improvement is probably due to not needing the Normal Roughness Buffer to work. This effect also has the benefit of working with alpha transparent materials, where as the built in effect ignores them.

## Instructions
1. Copy addons folder to Godot Project folder.
2. Add a new WorldEnviroment node.
3. Create a new Compositor. Under that, add a New SSAO.
4. Tweak settings to your liking.

## Atribution
Compositor Effect code was based off of the [Compositor Effect demo/tutorial](https://github.com/godotengine/godot-demo-projects/tree/master/compute/post_shader) for Godot 4.

Main shader code is based off of [this repository](https://github.com/scanberg/hbao).

Bilateral filter is based off of the one from [nvpro_samples gl_ssao](https://github.com/nvpro-samples/gl_ssao).

Additional improvements to the base algorithm where referenced from [NVIDIAGameWorks HBAOPlus](https://github.com/NVIDIAGameWorks/HBAOPlus), but no code was taken from there.

## TODO
1. Improve the ssao algorithm further.
2. Performance Improvements, such as interleaved rendering.
3. Improve the filtering pass.
4. Make compatible with the Mobile Renderer.
5. Likely bugfixes, I am not a graphics programmer.

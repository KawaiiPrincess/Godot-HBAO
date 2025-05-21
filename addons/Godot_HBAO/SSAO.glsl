#[compute]
#version 450

// Invocations in the (x, y, z) dimension
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(rgba16f, set = 0, binding = 0) uniform image2D color_image;
layout(set = 0, binding = 1) uniform sampler2D depth_texture;
layout(rgba16f, set = 0, binding = 2) uniform image2D blur_image;
layout(set = 0, binding = 3) uniform sampler2D noise_texture;

layout(set=2, binding=0) uniform uniformBuffer {
    mat4 inv_proj;
    mat4 proj;
} mat;


layout(set=3, binding=0) uniform SceneBuffer {
    float Bias;
    float Strength;
    float Radius;
    float Sharpness;
    float blur_radius;
    float Directions;
    float Samps;
    float LargeStrength;
} Scene;

// Our push constant
layout(push_constant, std430) uniform Params {
    vec2 raster_size;
    vec2 reserved;
} params;

const float PI = 3.14159265;
vec2 AORes = params.raster_size;
vec2 InvAORes = vec2(1.0/AORes.x, 1.0/AORes.y);

float fov = 2.0 * atan(1.0f / -mat.inv_proj[1][1] );
float scale = 1.0 / tan(fov * 0.5) * (AORes.y / AORes.x);

float AOStrength_small = Scene.Strength;
float AOStrength_large = Scene.LargeStrength;
float R = Scene.Radius;
float R2 = R*R;
float NegInvR2 = - 1.0 / (R*R);
float TanBias = tan(Scene.Bias * PI / 180.0);
float MaxRadiusPixels = 100.0;
float g_Sharpness = Scene.Sharpness;

int NumDirections = int(Scene.Directions);
int NumSamples = int(Scene.Samps);

int KERNEL_RADIUS = int(Scene.blur_radius);

vec2 hash22(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return fract((p3.xx+p3.yz)*p3.zy);

}

void lineardepth(vec2 uv, float depth, mat4 invmatrix, inout float linear_depth) {
    vec3 ndc = vec3(uv * 2.0 - 1.0, depth);

    vec4 view = invmatrix * vec4(ndc, 1.0);
    view.xyz /= view.w;
    linear_depth = view.z ;
}

vec4 boxFilter(ivec2 fragCoord, ivec2 res, int hw)
{
    vec2 uv = fragCoord/res;

    vec4 color = vec4(0);
    for (int u = -hw; u <= hw; ++u) {
        for (int v = -hw; v <= hw; ++v) {
            ivec2 bluruv = ivec2((vec2(fragCoord) + vec2(u,v)));
            color += imageLoad(blur_image, bluruv);
        }
    }
    color /= float((2*hw+1)*(2*hw+1));
    return color;
}

float TanToSin(float x)
{
    return x * inversesqrt(x*x + 1.0);
}

float InvLength(vec2 V)
{
    return inversesqrt(dot(V,V));
}

float Tangent(vec3 P, vec3 S)
{
    return -(P.z - S.z) * InvLength(S.xy - P.xy);
}

float BiasedTangent(vec3 V)
{
    return V.z * InvLength(V.xy) + TanBias;
}

float Length2(vec3 V)
{
    return dot(V,V);
}


vec3 MinDiff(vec3 P, vec3 Pr, vec3 Pl)
{
    vec3 V1 = Pr - P;
    vec3 V2 = P - Pl;
    return (Length2(V1) < Length2(V2)) ? V1 : V2;
}

float Falloff(float d2)
{
    return d2 * NegInvR2 + 1.0f;
}

vec2 SnapUVOffset(vec2 uv)
{
    return round(uv * AORes) * InvAORes;
}

vec3 UVToViewSpace(vec2 uv, float z)
{
    return vec3(uv, z);
}

vec3 GetViewPos(vec2 uv)
{
    float z = (texture(depth_texture, uv).r);
    lineardepth(uv,z,mat.proj,z);
    return vec3(uv, z);
}


vec2 RotateDirections(vec2 Dir, vec2 CosSin)
{
    return vec2(Dir.x*CosSin.x - Dir.y*CosSin.y,
                Dir.x*CosSin.y + Dir.y*CosSin.x);
}

vec3 getVector(vec2 p) {
    // normalized pixel coordinates
    vec2 q = p;
    // depth
    float depth = texture(depth_texture, p).r;
    //lineardepth(p,depth,mat.proj,depth);
    float t = depth;
    return vec3(q, t); // 3d vector
}

vec3 calcNormal(vec2 p) {
    vec3 w = getVector(p); // center vector
    vec3 u = getVector(p+vec2(0.0,0.001)); // top vector
    vec3 v = getVector(p+vec2(0.001,0.0)); // right vector

    return normalize(cross(w-u,w-v)); // normal
}

float HorizonOcclusion(	vec2 TexCoord,
                        vec2 deltaUV,
                        vec3 P,
                        vec3 dPdu,
                        vec3 dPdv,
                        float randstep)
{
    float ao = 0;

    // Offset the first coord with some noise
    vec2 uv = TexCoord + SnapUVOffset(randstep*deltaUV);
    deltaUV = SnapUVOffset( deltaUV );

    // Calculate the tangent vector
    vec3 T = deltaUV.x * dPdu + deltaUV.y * dPdv;

    // Get the angle of the tangent vector from the viewspace axis
    float tanH = BiasedTangent(T);
    float sinH = TanToSin(tanH);

    float tanS;
    float d2;
    vec3 S;

        uv += deltaUV;
        S = GetViewPos(uv);
        tanS = Tangent(P, S);
        d2 = Length2(S - P);

        // Is the sample within the radius and the angle greater?
        if(d2 < R2 && tanS > tanH)
        {

            float sinS = TanToSin(tanS);
            // Apply falloff based on the distance
            ao += (Falloff(d2) * (sinS - sinH));

            tanH = tanS;
            sinH = sinS;
        }


    return ao;
}

void ComputeSteps(inout vec2 stepSizeUv, inout float numSteps, float rayRadiusPix, float rand)
{
    // Avoid oversampling if numSteps is greater than the kernel radius in pixels
    numSteps = min(NumSamples, rayRadiusPix);

    // Divide by Ns+1 so that the farthest samples are not fully attenuated
    float stepSizePix = rayRadiusPix / (numSteps + 1);

    // Clamp numSteps if it is greater than the max kernel footprint
    float maxNumSteps = MaxRadiusPixels / stepSizePix;
    if (maxNumSteps < numSteps)
    {
        // Use dithering to avoid AO discontinuities
        numSteps = floor(maxNumSteps + rand);
        numSteps = max(numSteps, 1);
        stepSizePix = MaxRadiusPixels / numSteps;
    }

    // Step size in uv space
    stepSizeUv = stepSizePix * InvAORes;
}


void BlurFunction(vec2 uv, ivec2 fragcoord, float r, vec4 center_c, float center_d, inout float w_total, inout vec4 c_total)
{
    vec4  c = imageLoad( blur_image, fragcoord );
    float d = texture( depth_texture, uv).x;
    lineardepth(uv, d, mat.proj, d);
    //d = -d;

    const float BlurSigma = float(KERNEL_RADIUS) * 0.5;
    const float BlurFalloff = 1.0 / (2.0*BlurSigma*BlurSigma);

    float ddiff = (d - center_d) * g_Sharpness;
    float w = exp2(-r*r*BlurFalloff - ddiff*ddiff);
    w_total += w;

    c_total += c*w;
}


// The code we want to execute in each invocation
void main() {
    ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = ivec2(params.raster_size);

    // Prevent reading/writing out of bounds.
    if (uv.x >= size.x || uv.y >= size.y) {
        return;
    }

    // Read from our color buffer.
    vec4 color = imageLoad(color_image, uv);

    vec2 depth_uv = (vec2(uv) + 0.5) / size;

    vec2 noise_uv = hash22(depth_uv);
    vec3 randnoise = texture(noise_texture, noise_uv).rgb;

    float numDirections = NumDirections;
    vec3 P, Pr, Pl, Pt, Pb;
    P 	= GetViewPos(depth_uv);

    Pr 	= GetViewPos(depth_uv + vec2( InvAORes.x, 0));
    Pl 	= GetViewPos(depth_uv + vec2(-InvAORes.x, 0));
    Pt 	= GetViewPos(depth_uv + vec2( 0, InvAORes.y));
    Pb 	= GetViewPos(depth_uv + vec2( 0,-InvAORes.y));

    vec3 dPdu = MinDiff(P, Pr, Pl);
    vec3 dPdv = MinDiff(P, Pt, Pb);

    vec2 rayRadiusUV = vec2(0.5 * R * scale / -P.z);
    float rayRadiusPix = rayRadiusUV.x * size.x;

    float occlusion_small = 0.0;
    float occlusion_large = 0.0;

    float numSteps;
    vec2 stepSizeUV;

    float alpha = 2.0 * PI / numDirections;

    ComputeSteps(stepSizeUV, numSteps, rayRadiusPix, randnoise.z);

    for(float d = 0; d < numDirections; ++d) {

        float theta = alpha * d;

        vec2 dir = RotateDirections(vec2(cos(theta), sin(theta)), randnoise.xy);
        vec2 deltaUV = dir * stepSizeUV;

        occlusion_small += HorizonOcclusion(depth_uv,deltaUV,P,dPdu,dPdv,randnoise.z);

        for(float s = 1; s <= numSteps; ++s) {
            occlusion_large += HorizonOcclusion(depth_uv,deltaUV,P,dPdu,dPdv,randnoise.z);
            }
        }

    float ao_final = (occlusion_small * AOStrength_small / numDirections) + (occlusion_large * AOStrength_large * 0.5);
    ao_final /= (numDirections * numSteps);

    ao_final = clamp(1.0 - ao_final,0.0,1.0);

    float sky = texture(depth_texture, depth_uv).r;
    sky *= 4000.0;
    sky = clamp(sky,0.0,1.0);
    sky = 1.0 - sky;

    vec4 blur = vec4(ao_final,ao_final,ao_final,1.0);

    imageStore(blur_image, uv, blur);

    float flip = -1.0;
    float w_total = 1.0;
    vec4 final_color = blur;
    vec4 c_total = blur;
        for (float r = 1; r <= KERNEL_RADIUS; ++r)
        {
            vec2 direction = (int(r) % 2 == 0) ? vec2(flip * (1/AORes.x),0) : vec2(0,flip * (1/AORes.y));
            vec2 blur_uv = depth_uv - direction * r;
            ivec2 blur_fragcoord = ivec2( blur_uv * size);
            BlurFunction(blur_uv, blur_fragcoord, r, blur, P.z, w_total, c_total);
            flip = (int(r) % 2 == 0) ? 1.0 : -1.0;
        }
    final_color = c_total/w_total;

    final_color = clamp(final_color,0.0,1.0) + sky;
    final_color = clamp(final_color,0.0,1.0);
    final_color *= color;

    // Write back to our color buffer.
    imageStore(color_image, uv, final_color);
}

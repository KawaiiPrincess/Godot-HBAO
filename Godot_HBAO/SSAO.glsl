#[vertex]

#version 450

layout(location = 0) out vec2 uv_interp;

void main() {
    vec2 vertex_base;
    if (gl_VertexIndex == 0) {
        vertex_base = vec2(-1.0, -1.0);
    } else if (gl_VertexIndex == 1) {
        vertex_base = vec2(-1.0, 3.0);
    } else {
        vertex_base = vec2(3.0, -1.0);
    }
    gl_Position = vec4(vertex_base, 0.0, 1.0);
    uv_interp = clamp(vertex_base, vec2(0.0, 0.0), vec2(1.0, 1.0)) * 2.0; // saturate(x) * 2.0
}

#[fragment]

#version 450

layout(set = 0, binding = 0) uniform sampler2D depth_texture;

layout(location = 0) in vec2 uv_interp;
layout(location = 0) out vec4 frag_color;

layout(set=2, binding=0) uniform uniformBuffer {
    mat4 inv_proj;
    mat4 proj;
} mat;


layout(set=3, binding=0) uniform SceneBuffer {
    float Bias;
    float Strength;
    float Radius;
    float Sharpness;
    float Power;
    float null1;
    float null2;
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

float UVToViewA = -2.0f * (1.0 / scale);
float UVToViewB =  1.0f * (1.0 / scale);

float AOStrength_small = Scene.Strength;
float AOStrength_large = Scene.LargeStrength;
float R = Scene.Radius;
float R2 = R*R;
float NegInvR2 = - 1.0 / (R*R);
float TanBias = tan(Scene.Bias * PI / 180.0);
float MaxRadiusPixels = 50.0;

const int NumDirections = 8;
const int NumSamples = 4;

float IGN(vec2 p)
{
    vec3 magic = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract( magic.z * fract(dot(p,magic.xy)) );
}

void lineardepth(vec2 uv, float depth, mat4 invmatrix, inout float linear_depth) {
    vec3 ndc = vec3(uv * 2.0 - 1.0, depth);

    vec4 view = invmatrix * vec4(ndc, 1.0);
    view.xyz /= view.w;
    linear_depth = view.z ;
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
    uv = UVToViewA * uv + UVToViewB;
    return vec3(uv * z, z);
}

vec3 GetViewPos(vec2 uv)
{
    float z = (texture(depth_texture, uv).r);
    lineardepth(uv,z,mat.proj,z);
    return UVToViewSpace(uv, z);
}

vec2 RotateDirections(vec2 Dir, vec2 CosSin)
{
    return vec2(Dir.x*CosSin.x - Dir.y*CosSin.y,
                Dir.x*CosSin.y + Dir.y*CosSin.x);
}


vec2 orientate(vec2 a, vec3 b) {
    return a * sign(dot(vec3(a,1.0),b));
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
        ao = (Falloff(abs(d2)) * (sinS - sinH));

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

vec2 encode_depth(float value) {
    vec2 kEncodeMul = vec2(1.0f, 255.0f);
    float kEncodeBit = 1.0f/255.0f;
    vec2 color = kEncodeMul*value;
    color = fract(color);
    color.x -= color.y * kEncodeBit;
    return color;
}

// The code we want to execute in each invocation
void main() {
    vec2 size = params.raster_size;

    vec2 depth_uv = uv_interp * 2.0;

    float AONoise = IGN(uv_interp * size);

    vec3 P, Pr, Pl, Pt, Pb;

    P 	= GetViewPos(depth_uv);

    vec2 depth_encode = encode_depth((P.z + (P.z / 2.0)) / 256.0);

    Pr 	= GetViewPos(depth_uv + vec2( InvAORes.x, 0));
    Pl 	= GetViewPos(depth_uv + vec2(-InvAORes.x, 0));
    Pt 	= GetViewPos(depth_uv + vec2( 0, InvAORes.y));
    Pb 	= GetViewPos(depth_uv + vec2( 0,-InvAORes.y));

    vec3 dPdu = MinDiff(P, Pr, Pl);
    vec3 dPdv = MinDiff(P, Pt, Pb) * (AORes.y * InvAORes.x);

    vec3 normal = normalize(cross(dPdu,dPdv));

    vec2 rayRadiusUV = vec2(0.25 * R * scale / -P.z);
    float rayRadiusPix = rayRadiusUV.x * AORes.x;

    float occlusion_small = 0.0;
    float occlusion_large = 0.0;

    float alpha = 2.0 * PI / NumDirections;

    float numSteps;
    vec2 stepSizeUV;

    ComputeSteps(stepSizeUV,numSteps, rayRadiusPix, AONoise);

    for(float d = 0; d < NumDirections; ++d) {

        float theta = alpha * d;

        vec2 dir = RotateDirections(vec2(cos(theta), sin(theta)), vec2(AONoise));
        vec2 deltaUV = orientate(dir * stepSizeUV,normal);


        occlusion_small += HorizonOcclusion(depth_uv,deltaUV,P,dPdu,dPdv,AONoise);

        for(float s = 1; s <= NumSamples; ++s) {

            occlusion_large += HorizonOcclusion(depth_uv,deltaUV,P,dPdu,dPdv,AONoise);
        }
    }

    float ao_final = max(occlusion_small * AOStrength_small,0.0) + max(occlusion_large * AOStrength_large,0.0);
    ao_final /= (NumDirections * NumSamples);

    ao_final = clamp(pow(1.0 - ao_final, Scene.Power),0.0,1.0);


    vec4 blur = vec4(depth_encode.r,depth_encode.g,1.0,ao_final);

    frag_color = blur;
}

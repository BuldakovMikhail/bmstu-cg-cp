#version 330 core

layout (location = 0) out vec4 fragColor;

uniform vec2 u_resolution;
uniform vec3 u_sun_pos;

// uniform sampler2D u_weatherMap;

const vec3 camPos = vec3(0, 6400, 0);
const float maxCloud = 6435;
const float minCloud = 6415;

const float pRad = 6400;

const float iSteps = 32;
const float jSteps = 6;

const float FOV = 1; 

vec2 rsi(in vec3 ro, in vec3 rd, float sr){
	float a = dot(rd, rd);
	float b = 2.0 * dot(ro, rd);
	float c = dot(ro,ro) - (sr * sr);

	float d = b * b - 4.0 * a * c;

	if (d < 0.0) return vec2(1e5, -1e5);

	return vec2(
		(-b - sqrt(d))/(2.0*a),
        (-b + sqrt(d))/(2.0*a));
}

mat3 getCam(vec3 ro, vec3 lookAt) {
    vec3 camF = normalize(vec3(lookAt - ro));
    vec3 camR = normalize(cross(vec3(0, 1, 0), camF));
    vec3 camU = cross(camF, camR);
    return mat3(camR, camU, camF);
}

float cloudSampleDensity(vec3 position){
	return position.x;
}

float HenyeyGreenstein(float g, float mu) {
  float gg = g * g;
	return (1.0 / (4.0 * PI))  * ((1.0 - gg) / pow(1.0 + gg - 2.0 * g * mu, 1.5));
}

vec3 rayMarching(in vec2 uv){
	vec3 ro = camPos;
    vec3 lookAt = ro + u_sun_pos;
	vec3 rd = getCam(ro, lookAt) * normalize(vec3(uv, FOV));
	
	vec2 farInter = rsi(ro, rd, maxCloud);
	vec2 closeInter = rsi(ro, rd, minCloud);

	float stepT = (farInter.y - closeInter.y) / iSteps; 

	vec3 iPos = ro + rd * closeInter.y;

	float odCloud = 0;



	for (int i = 0; i < iSteps; ++i){
	
		iPos += rd * stepT;

		odCloud += cloudSampleDensity(iPos); 
	}

	return vec3(odCloud);
}

void main()
{
	vec2 uv = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;

    vec3 col = rayMarching(uv);;

    col *= vec3(2, 1.8, 2); // color correction

    // gamma correction
    vec3 tunedColor=col/(1+col);
    tunedColor = pow(tunedColor, vec3(1.0/2.2));
    fragColor = vec4(tunedColor, 1);
}
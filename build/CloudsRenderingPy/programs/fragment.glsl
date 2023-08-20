#version 330 core

#define PI 3.1415

layout (location = 0) out vec4 fragColor;



	uniform float u_density;
	uniform float u_coverage;
	uniform float u_phaseInfluence;
	uniform float u_eccentrisy;

	uniform float u_phaseInfluence2;
	uniform float u_eccentrisy2;
	uniform float u_attenuation;
	uniform float u_attenuation2;
	uniform float u_sunIntensity;
	uniform float u_fog;
	uniform float u_ambient;



uniform vec2 u_resolution;
uniform float u_time;
uniform vec3 u_sun_pos;

uniform sampler2D u_weatherMap;


#include noise.glsl

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

float cloudGetHeight(vec3 position){
	return (position.z - minCloud) / (maxCloud - minCloud);
}

float cloudSampleDensity(vec3 position)
{
	// position.xz+=vec2(0.2f)*u_time;

	vec4 weather=textureLod(u_weatherMap, position.xz/4096.0f+vec2(0.2, 0.1), 0);
	float height=cloudGetHeight(position);
	
	float SRb=clamp(remap(height, 0, 0.07, 0, 1), 0, 1);
	float SRt=clamp(remap(height, weather.b*0.2, weather.b, 1, 0), 0, 1);
	float SA=SRt;
	
	float DRb=height*clamp(remap(height, 0, 0.15, 0, 1), 0, 1) + 0.5;
	float DRt=height*clamp(remap(height, 0.9, 1, 1, 0), 0, 1);
	float DA=weather.a*2*u_density; // Это ноль, но почему? drb 

	// У Drb и Srb - 0, вопрос какого хера? 
	
	vec4 noise = stackable3DNoise(position/48.0f);
	float final = remap(noise.x, (noise.y * 0.625 + noise.z*0.25 + noise.w * 0.125)-1.0, 1.0, 0.0, 1.0);
	float SNsample=final; 
	
	float WMc=max(weather.r, clamp(u_coverage-0.5, 0, 1)*weather.g*2);
	float d=clamp(remap(SNsample*SA, 1-u_coverage*WMc, 1, 0, 1), 0, 1)*DA;
	
	return d;
}

float HenyeyGreenstein(float g, float mu) {
  float gg = g * g;
	return (1.0 / (4.0 * PI))  * ((1.0 - gg) / pow(1.0 + gg - 2.0 * g * mu, 1.5));
}

float cloudSampleDirectDensity(vec3 position, vec3 sunDir)
{
	//определяем размер шага
	float avrStep=(6435.0-6415.0)*0.01;
	float sumDensity=0.0;
	for(int i=0;i<4;i++)
	{
		float step=avrStep;
		//для последней выборки умножаем шаг на 6
		if(i==3)
			step=step*6.0;
		//обновляем позицию
		position+=sunDir*step;
		//получаем значение плотности, вызывая функцию, которая уже 
                //рассматривалась ранее
		float density=cloudSampleDensity(position)*step;
		sumDensity+=density;
	}
	return sumDensity;
}

vec4 mainMarching(vec3 ro, vec3 viewDir, vec3 sunDir, vec3 sunColor, vec3 ambientColor)
{
	vec2 t = rsi(ro, viewDir, minCloud);
	vec3 position = ro + viewDir * t.y;

	// crossRaySphereOutFar(vec3(0.0, 6400.0, 0.0), viewDir, vec3(0.0), 6415.0, position);
	
	float avrStep=(6435.0-6415.0)/64.0;
	
	vec3 color=vec3(0.0);
	float transmittance=1.0;
	float density = 0.0;

	for(int i=0;i<128;i++)
	{
		density += cloudSampleDensity(position)*avrStep;
		if(density>0.0)
		{
			float sunDensity=cloudSampleDirectDensity(position, sunDir);
			float mu=max(0.0, dot(viewDir, sunDir));
				
			float m11=u_phaseInfluence*HenyeyGreenstein(mu, u_eccentrisy);
			float m12=u_phaseInfluence2*HenyeyGreenstein(mu, u_eccentrisy2);
			float m2=exp(-u_attenuation*sunDensity);
			float m3=u_attenuation2*density;
			float light=u_sunIntensity*(m11+m12)*m2*m3;
		
			color+=sunColor*light*transmittance;
			transmittance*=exp(-u_attenuation*density);
		}
		position+=viewDir*avrStep;

		if(transmittance<0.05 || length(position)>6435.0)
			break;
	}
	
	// float blending=1.0-exp(-max(0.0, dot(viewDir, vec3(0.0,1.0,0.0)))*u_fog);
	float blending = 1;
	blending=blending*blending*blending;
	//return vec4(mix(ambientColor, color+ambientColor*u_ambient, blending), 1.0-transmittance);

	return vec4(vec3(density), 1);
}

void main()
{
	vec2 uv = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;

	vec3 ro = camPos;
    vec3 lookAt = ro + u_sun_pos;
	vec3 rd = getCam(ro, lookAt) * normalize(vec3(uv, FOV));

    // vec3 col = rayMarching(uv);

    // col *= vec3(2, 1.8, 2); // color correction

    // // gamma correction
    // vec3 tunedColor=col/(1+col);
    // tunedColor = pow(tunedColor, vec3(1.0/2.2));
    fragColor = mainMarching(ro, rd, normalize(u_sun_pos), vec3(1, 1, 1), vec3(0.3, 0.79, 1));
}
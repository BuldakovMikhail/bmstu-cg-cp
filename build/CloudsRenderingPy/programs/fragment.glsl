#version 330 core

#define PI 3.1415

layout (location = 0) out vec4 fragColor;

vec3 saturate3(vec3 x) {
  return clamp(x, vec3(0.0), vec3(1.0));
}

float saturate(float height){
	height -= 0.4;
    height *= 100;
    float v = 2 * 2 * 2 / (height * height  + 2 * 2);
    v *= 100;
    return clamp(v, 0, 1);
  
  // return 1;
}

const float AMBIENT_STRENGTH = 0.1;
const float CLOUD_LIGHT_MULTIPLIER = 50.0;
const vec3 EXTINCTION_MULT = vec3(0.8, 0.8, 1.0);
const float DUAL_LOBE_WEIGHT = 0.7;

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

uniform vec3 u_look_at;

uniform sampler2D u_weatherMap;
uniform sampler3D u_lfNoise;
uniform sampler3D u_hfNoise;


const vec3 camPos = vec3(0, 6400, 0);
const float maxCloud = 6435;
const float minCloud = 6415;

const float pRad = 6400;

const float iSteps = 32;
const float jSteps = 6;

const float FOV = 1; 

// float saturate(float x) {
//   return clamp(x, 0.0, 1.0);
// }

float remap(float value, float minValue, float maxValue, float newMinValue, float newMaxValue)
{
    return newMinValue+(value-minValue)/(maxValue-minValue)*(newMaxValue-newMinValue);
}

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

float cloudGetHeight(vec3 position, vec2 cloudMinMax){
	return (position.y - minCloud) / (maxCloud - minCloud);
}

float cloudSampleDensity(vec3 position, vec2 cloudMinMax)
{
	position.xz+=vec2(0.5)*u_time; 
	float base = texture(u_lfNoise, position / 48).r;
	float height = cloudGetHeight(position, cloudMinMax);
	
  float coverage = texture(u_weatherMap, position.xz / 480).r;
  float coff = saturate(height);

  base *= coff;

  float baseCloudWithCoverage = remap(base, 1-coverage, 1, 0, 1);
  // float baseCloudWithCoverage = base;
  baseCloudWithCoverage *= coverage;

  float hfFBM = texture(u_hfNoise, position / 48).r;
  float hfNoiseModifier = mix(hfFBM, 1 - hfFBM, clamp(height * 10, 0, 1));

  float finalCloud = remap(baseCloudWithCoverage, hfNoiseModifier * 0.2, 1, 0, 1);


	return max(finalCloud, 0);
}



float HenyeyGreenstein(float g, float mu) {
  float gg = g * g;
	return (1.0 / (4.0 * PI))  * ((1.0 - gg) / pow(1.0 + gg - 2.0 * g * mu, 1.5));
}

float cloudSampleDirectDensity(vec3 position, vec3 sunDir, vec2 cloudMinMax)
{
	//определяем размер шага
	float avrStep=(6435.0-6415.0)*0.01;
	float sumDensity=0.0;
	for(int i=0;i<4;i++)
	{
		float step=avrStep;
		//для последней выборки умножаем шаг на 6
		// if(i==3)
		// 	step=step*6.0;
		//обновляем позицию
		position+=sunDir*step;
		//получаем значение плотности, вызывая функцию, которая уже 
                //рассматривалась ранее
		float density=cloudSampleDensity(position, cloudMinMax)*step;
		sumDensity+=density;
	}
	return sumDensity;
}

vec3 atmosphere(vec3 r, vec3 r0, vec3 pSun, float iSun, float rPlanet, float rAtmos, vec3 kRlh, vec3 kMie, float shRlh, float shMie, float g) {
    // Normalize the sun and view directions.
    pSun = normalize(pSun);
    r = normalize(r);

    // Calculate the step size of the primary ray.
    vec2 p = rsi(r0, r, rAtmos);
    if (p.x > p.y) return vec3(0,0,0);
    p.y = min(p.y, rsi(r0, r, rPlanet).x);
    float iStepSize = (p.y - p.x) / float(iSteps);

    // Initialize the primary ray time.
    float iTime = 0.0;

    // Initialize accumulators for Rayleigh and Mie scattering.
    vec3 totalRlh = vec3(0,0,0);
    vec3 totalMie = vec3(0,0,0);

    // Initialize optical depth accumulators for the primary ray.
    float iOdRlh = 0.0;
    float iOdMie = 0.0;

    // Calculate the Rayleigh and Mie phases.
    float mu = dot(r, pSun);
    float mumu = mu * mu;
    float gg = g * g;
    float pRlh = 3.0 * PI / (16.0) * (1.0 + mumu);
    float pMie = 3.0 * PI / (8.0) * ((1.0 - gg) * (mumu + 1.0)) / (pow(1.0 + gg - 2.0 * mu * g, 1.5) * (2.0 + gg));

    // Sample the primary ray.
    for (int i = 0; i < iSteps; i++) {

        // Calculate the primary ray sample position.
        vec3 iPos = r0 + r * (iTime + iStepSize * 0.5);

        // Calculate the height of the sample.
        float iHeight = length(iPos) - rPlanet;

        // Calculate the optical depth of the Rayleigh and Mie scattering for this step.
        float odStepRlh = exp(-iHeight / shRlh) * iStepSize;
        float odStepMie = exp(-iHeight / shMie) * iStepSize;

        // Accumulate optical depth.
        iOdRlh += odStepRlh;
        iOdMie += odStepMie;

        // Calculate the step size of the secondary ray.
        float jStepSize = rsi(iPos, pSun, rAtmos).y / float(jSteps);

        // Initialize the secondary ray time.
        float jTime = 0.0;

        // Initialize optical depth accumulators for the secondary ray.
        float jOdRlh = 0.0;
        float jOdMie = 0.0;

        // Sample the secondary ray.
        for (int j = 0; j < jSteps; j++) {

            // Calculate the secondary ray sample position.
            vec3 jPos = iPos + pSun * (jTime + jStepSize * 0.5);

            // Calculate the height of the sample.
            float jHeight = length(jPos) - rPlanet;

            // Accumulate the optical depth.
            jOdRlh += exp(-jHeight / shRlh) * jStepSize;
            jOdMie += exp(-jHeight / shMie) * jStepSize;

            // Increment the secondary ray time.
            jTime += jStepSize;
        }

        // Calculate attenuation.
        vec3 attnMie = exp(-kMie * (iOdMie + jOdMie));
		vec3 attnRlh = exp(-kRlh * (iOdRlh + jOdRlh));

        // Accumulate scattering.
        totalRlh += odStepRlh * attnRlh;
        totalMie += odStepMie * attnMie;

        // Increment the primary ray time.
        iTime += iStepSize;

    }

    // Calculate and return the final color.

    return iSun * (pRlh * kRlh * totalRlh + pMie * kMie * totalMie);
    //return iSun * (pMie * kMie * totalMie);
}

float DualHenyeyGreenstein(float g, float costh) {
  return mix(HenyeyGreenstein(-g, costh), HenyeyGreenstein(g, costh), DUAL_LOBE_WEIGHT);
}

float PhaseFunction(float g, float costh) {
  return DualHenyeyGreenstein(g, costh);
}

vec3 MultipleOctaveScattering(float density, float mu) {
  float attenuation = 0.2;
  float contribution = 0.2;
  float phaseAttenuation = 0.5;

  float a = 1.0;
  float b = 1.0;
  float c = 1.0;
  float g = 0.85;
  const float scatteringOctaves = 4.0;
  
  vec3 luminance = vec3(0.0);

  for (float i = 0.0; i < scatteringOctaves; i++) {
    float phaseFunction = PhaseFunction(0.3 * c, mu);
    vec3 beers = exp(-density * EXTINCTION_MULT * a);

    luminance += b * phaseFunction * beers;

    a *= attenuation;
    b *= contribution;
    c *= (1.0 - phaseAttenuation);
  }
  return luminance;
}


vec3 calculateLightEnergy(vec3 position, vec3 sunDir, float mu, vec2 cloudMinMax) {
  
  float density = cloudSampleDirectDensity(position, sunDir, cloudMinMax)* u_attenuation2;
  vec3 beersLaw = MultipleOctaveScattering(density, mu);
  vec3 powder = 1.0 - exp(-density * 2.0 * EXTINCTION_MULT);

return beersLaw * mix(2.0 * powder, vec3(1.0), remap(mu, -1.0, 1.0, 0.0, 1.0));
}

vec4 mainMarching(vec3 ro, vec3 viewDir, vec3 sunDir, vec3 sunColor, vec3 ambientColor)
{
	vec2 t = rsi(ro, viewDir, minCloud);
	vec3 position = ro + viewDir * t.y;

	vec2 t2 = rsi(ro, viewDir, maxCloud);
	vec3 position2 = ro + viewDir * t2.y;

  vec3 atmoColor = atmosphere(
  viewDir,
  vec3(0, 6371000, 0),
  normalize(u_sun_pos),
  2.5,
  6371000,
  6471000,
  vec3(5.8e-6, 13.5e-6, 33.1e-6),
  vec3(3e-6, 3e-6, 3e-6),
  8e3,
  1.2e3,
  0.996
  );

  // if (position.y > position2.y)
  //   return vec4(atmoColor, 1);

	float avrStep = (maxCloud - minCloud) / 64;

	vec2 cloudMinMax;
	cloudMinMax.x = position.z;
	cloudMinMax.y = position2.z;

	// crossRaySphereOutFar(vec3(0.0, 6400.0, 0.0), viewDir, vec3(0.0), 6415.0, position);
	
	vec3 iPos = position;

	float density = 0;
	// float bl = 1;

	float mu = dot(viewDir, sunDir);

	float l = 0;
	// vec3 color = vec3(0);

	vec3 transmittance = vec3(1);
	vec3 scattering = vec3(0);

	vec3 sunLightColor = vec3(1.0);
    vec3 sunLight = sunLightColor * CLOUD_LIGHT_MULTIPLIER;
	vec3 ambient = vec3(AMBIENT_STRENGTH * sunLightColor) * u_ambient;

	for (int i = 0; i < 128; ++i){
		
		if (length(iPos) > maxCloud)
			break;
		density = cloudSampleDensity(iPos, cloudMinMax);

    if (density > 0.01){
      vec3 luminance = ambient + sunLight * calculateLightEnergy(iPos, sunDir, mu, cloudMinMax);
		vec3 ttransmittance = exp(-density * avrStep * EXTINCTION_MULT * u_attenuation);
		vec3 integScatt = density * (luminance - luminance * ttransmittance) / density;

		scattering += transmittance * integScatt;
		transmittance *= ttransmittance;  
      if (length(transmittance) <= 0.01) {
              transmittance = vec3(0.0);
              break;
      }
    }


		iPos += viewDir * avrStep;
	}



	
	transmittance = saturate3(transmittance);
	
	
	
	vec3 color = atmoColor.xyz * transmittance + scattering;

	return vec4(color, 1);

	// float bl = exp(-0.1 * density);
	// return vec4(vec3(density / (density + 1)), 1);
	// return vec4(vec3(l / (l + 1)), 1);
	//return vec4(mix(color, atmoColor, transmittance), 1);
	// return vec4(vec3(transmittance / (transmittance + 1)), 1);
}	

void main()
{
	vec2 uv = (2.0 * gl_FragCoord.xy - u_resolution.xy) / u_resolution.y;

	vec3 ro = camPos;
    vec3 lookAt = ro + u_look_at;
	vec3 rd = getCam(ro, lookAt) * normalize(vec3(uv, FOV));

    vec4 col = mainMarching(ro, rd, normalize(u_sun_pos), vec3(1, 1, 1), vec3(0.3, 0.79, 1));

    // col *= vec3(2, 1.8, 2); // color correction

    // gamma correction
    vec3 tunedColor=col.rgb/(1+col.rgb);
    tunedColor = pow(tunedColor, vec3(1.0/2.2));
    fragColor = vec4(tunedColor, col.a);
}
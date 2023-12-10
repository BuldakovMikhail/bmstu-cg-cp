const vec3 EXTINCTION_MULT = vec3(0.8, 0.8, 1.0);
const float CLOUD_LIGHT_MULTIPLIER = 50.0;

vec4 mainMarching(vec3 ro, vec3 viewDir, vec3 sunDir, vec3 sunColor, vec3 ambientColor)
{
	vec2 t = rsi(ro, viewDir, minCloud);
	vec3 position = ro + viewDir * t.y;

  	vec3 atmoColor = getAtmoColor(viewDir);

	float avrStep = (maxCloud - minCloud) / 64;

	vec3 iPos = position;

	float density = 0;

	float mu = dot(viewDir, sunDir);

	vec3 transmittance = vec3(1);
	vec3 scattering = vec3(0);

	vec3 sunLightColor = vec3(1.0);
  	vec3 sunLight = sunLightColor * CLOUD_LIGHT_MULTIPLIER;
	vec3 ambient = vec3(AMBIENT_STRENGTH * sunLightColor) * u_ambient;

	for (int i = 0; i < 128; ++i){
		if (length(iPos) > maxCloud)
				break;
		density = cloudSampleDensity(iPos);
		
		if (density > 0.01){
			vec3 luminance = ambient + sunLight * calculateLightEnergy(iPos, sunDir, mu);
				vec3 ttransmittance = exp(-density * avrStep * EXTINCTION_MULT * u_attenuation);
				vec3 integScatt = luminance * (1 - ttransmittance);
				scattering += transmittance * integScatt;
				transmittance *= ttransmittance;  
				
			if (length(transmittance) <= 0.01) {
					transmittance = vec3(0.0);
					break;
			}
		}

		iPos += viewDir * avrStep;
	}
	
  	vec3 color = atmoColor.xyz * transmittance + scattering;

	return vec4(color, 1);
}	
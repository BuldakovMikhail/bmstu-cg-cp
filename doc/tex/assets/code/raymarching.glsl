const vec3 EXTINCTION_MULT = vec3(0.8, 0.8, 1.0);
const float CLOUD_LIGHT_MULTIPLIER = 50.0;

vec4 mainMarching(vec3 ro, vec3 viewDir, vec3 sunDir, vec3 sunColor, vec3 ambientColor)
{
	vec2 t = rsi(ro, viewDir, minCloud);
	vec3 position = ro + viewDir * t.y;

	vec2 t2 = rsi(ro, viewDir, maxCloud);
	vec3 position2 = ro + viewDir * t2.y;

	float avrStep = (maxCloud - minCloud) / 64;

	vec2 cloudMinMax;
	cloudMinMax.x = position.y;
	cloudMinMax.y = position2.y;
	
	vec3 iPos = position;

	float mu = dot(viewDir, sunDir);

	vec3 transmittance = vec3(1);
	vec3 scattering = vec3(0);

    vec3 sunLight = sunColor * CLOUD_LIGHT_MULTIPLIER;
	vec3 ambient = vec3(AMBIENT_STRENGTH * sunColor) * u_ambient;

	for (int i = 0; i < 128; ++i){
		if (length(iPos) > maxCloud) break;

		float density = cloudSampleDensity(iPos, cloudMinMax);

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
	vec3 color = ambientColor.xyz * transmittance + scattering;

	return vec4(color, 1);
}	
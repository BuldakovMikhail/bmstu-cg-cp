const float DUAL_LOBE_WEIGHT = 0.7;

float henyeyGreenstein(float g, float mu) {
    float gg = g * g;
	return (1.0 / (4.0 * PI))  * ((1.0 - gg) / pow(1.0 + gg - 2.0 * g * mu, 1.5));
}

float dualhenyeyGreenstein(float g, float costh) {
  return mix(henyeyGreenstein(-g, costh), henyeyGreenstein(g, costh), DUAL_LOBE_WEIGHT);
}

float phaseFunction(float g, float costh) {
  return dualhenyeyGreenstein(g, costh);
}

float cloudSampleDirectDensity(vec3 position, vec3 sunDir)
{
	float avrStep=(6435.0-6415.0)*0.01;
	float sumDensity=0.0;
	for(int i=0;i<4;i++)
	{
		float step=avrStep;
		if(i==3)
			step=step*6.0;

		position+=sunDir*step;
		float density=cloudSampleDensity(position)*step;
		sumDensity+=density;
	}
	return sumDensity;
}

vec3 calculateLightEnergy(vec3 position, vec3 sunDir, float mu) {
  
    float density = cloudSampleDirectDensity(position, sunDir)* u_attenuation2;
    vec3 beersLaw = exp(-density * EXTINCTION_MULT) * phaseFunction(u_eccentrisy2, mu);
    vec3 powder = 1.0 - exp(-density * 2.0 * EXTINCTION_MULT);

    return beersLaw * mix(2.0 * powder, vec3(1.0), remap(mu, -1.0, 1.0, 0.0, 1.0));
}
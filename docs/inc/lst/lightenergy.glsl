float HenyeyGreenstein(float g, float mu) {
    float gg = g * g;
	return (1.0 / (4.0 * PI))  * ((1.0 - gg) / pow(1.0 + gg - 2.0 * g * mu, 1.5));
}

float cloudSampleDirectDensity(vec3 position, vec3 sunDir, vec2 cloudMinMax)
{
	float avrStep=(cloudMinMax.y - cloudMinMax.x)*0.01;
	float sumDensity=0.0;

	for(int i=0;i<4;i++)
	{
		float step=avrStep;

		if (i==3)
		    step=step*6.0;
		
		position+=sunDir*step;
		float density=cloudSampleDensity(position, cloudMinMax)*step;
		sumDensity+=density;
	}
	return sumDensity;
}

vec3 calculateLightEnergy(vec3 position, vec3 sunDir, float mu, vec2 cloudMinMax) {
  
    float density = cloudSampleDirectDensity(position, sunDir, cloudMinMax)* u_attenuation2;
    vec3 beersLaw = exp(-density * EXTINCTION_MULT * a) * HenyeyGreenstein(u_eccentrisy2, mu);
    vec3 powder = 1.0 - exp(-density * 2.0 * EXTINCTION_MULT);

    return beersLaw * mix(2.0 * powder, vec3(1.0), remap(mu, -1.0, 1.0, 0.0, 1.0));
}
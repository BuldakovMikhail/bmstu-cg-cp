float cloudGetHeight(vec3 position, vec2 cloudMinMax){
	return (position.y - cloudMinMax.x) / (cloudMinMax.y - cloudMinMax.x);
}

float saturate(float height){
	height -= 0.4;
    height *= 100;
    float v = 2 * 2 * 2 / (height * height  + 2 * 2);
    v *= 100;
    
    return clamp(v, 0, 1);
}

float remap(float value, float minValue, float maxValue, float newMinValue, float newMaxValue)
{
    return newMinValue+(value-minValue) /
        (maxValue-minValue)*(newMaxValue-newMinValue);
}

float cloudSampleDensity(vec3 position, vec2 cloudMinMax)
{
	position.xz+=vec2(0.2)*u_time; 
	float base = texture(u_lfNoise, position / 48).r;
	float height = cloudGetHeight(position, cloudMinMax);
	
    float coverage = texture(u_weatherMap, position.xz / 480).r;
    float coff = saturate(height);

    base *= coff;

    float baseCloudWithCoverage = remap(base, 1-coverage, 1, 0, 1);

    baseCloudWithCoverage *= coverage;

    float hfFBM = texture(u_hfNoise, position / 48).r;
    float hfNoiseModifier = mix(hfFBM, 1 - hfFBM, clamp(height * 10, 0, 1));

    float finalCloud = remap(baseCloudWithCoverage, hfNoiseModifier * 0.2, 1, 0, 1);


	return max(finalCloud, 0);
}
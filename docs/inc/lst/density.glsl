const float maxCloud = 6435;
const float minCloud = 6415;

float cloudGetHeight(vec3 position){
	return (position.y - minCloud) / (maxCloud - minCloud);
}

float remap(float value, float minValue, float maxValue, float newMinValue, float newMaxValue)
{
    return newMinValue+(value-minValue) /
        (maxValue-minValue)*(newMaxValue-newMinValue);
}

float cloudSampleDensity(vec3 position)
{
	position.xz+=vec2(0.5)*u_time; 

    vec4 weather=texture(u_weatherMap, position.xz/480+vec2(0.2, 0.1));
	float height=cloudGetHeight(position);
	
	float SRb=clamp(remap(height, 0, 0.07, 0, 1), 0, 1);
	float SRt=clamp(remap(height, weather.b*0.2, weather.b, 1, 0), 0, 1);
	float SA=SRb*SRt;
	
	float DRb=height*clamp(remap(height, 0, 0.15, 0, 1), 0, 1);
	float DRt=height*clamp(remap(height, 0.9, 1, 1, 0), 0, 1);
	float DA=DRb*DRt*weather.a*2*u_density;
	
	float SNsample=texture(u_lfNoise, position/48.0f).x*0.85f+texture(u_hfNoise, position/48).x*0.15f; 
	
	float WMc=max(weather.r, clamp(u_coverage-0.5, 0, 1)*weather.g*2);
	float d=clamp(remap(SNsample*SA, 1-u_coverage*WMc, 1, 0, 1), 0, 1)*DA;
	
	return d;
}

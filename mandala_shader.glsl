#version 150

uniform sampler2D tex0;
uniform float time;
uniform vec2 resolution;
uniform float rotation;
uniform float connection_strength[12][12];  // Connection strengths matrix
uniform float voice_states[12][768];       // Voice states matrix
uniform float projections[12];             // Voice projections
uniform float reflections[12];             // Reflection magnitudes

in vec2 texcoord;
out vec4 outputColor;

const float PI = 3.14159265359;

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

// Calculate mandala element position
vec2 getElementPosition(float angle, float radius, float layer) {
    float x = cos(angle) * radius;
    float y = sin(angle) * radius;
    return vec2(x, y);
}

void main() {
    vec2 uv = texcoord * 2.0 - 1.0;
    vec2 center = vec2(0.0);
    float dist = length(uv - center);
    
    // Base color
    vec3 color = vec3(0.0);
    
    // Number of layers and elements
    int numLayers = 5;
    int elementsPerLayer = 12;
    
    // Overall rotation
    float globalRotation = rotation + time * 0.1;
    
    // Draw mandala elements
    for(int layer = 0; layer < numLayers; layer++) {
        float layerRadius = 0.2 + float(layer) * 0.15;
        
        for(int element = 0; element < elementsPerLayer; element++) {
            float angle = float(element) * (2.0 * PI / float(elementsPerLayer)) + globalRotation;
            vec2 elementPos = getElementPosition(angle, layerRadius, float(layer));
            
            // Calculate element color based on voice states and projections
            float projection = projections[element];
            float reflection = reflections[element];
            
            // Map projection to hue
            float hue = mod(projection * 0.5 + float(layer) * 0.1, 1.0);
            float saturation = 0.8;
            float value = 0.8 + reflection * 0.2;
            
            vec3 elementColor = hsv2rgb(vec3(hue, saturation, value));
            
            // Draw element
            float elementSize = 0.05 + reflection * 0.02;
            float elementDist = length(uv - elementPos);
            
            if(elementDist < elementSize) {
                // Add connection lines
                for(int other = 0; other < elementsPerLayer; other++) {
                    if(other != element) {
                        float strength = connection_strength[element][other];
                        if(strength > 0.1) {
                            float otherAngle = float(other) * (2.0 * PI / float(elementsPerLayer)) + globalRotation;
                            vec2 otherPos = getElementPosition(otherAngle, layerRadius, float(layer));
                            
                            // Draw connection line
                            float lineWidth = 0.002 * strength;
                            float lineDist = abs(dot(normalize(otherPos - elementPos), uv - elementPos));
                            if(lineDist < lineWidth && 
                               dot(otherPos - elementPos, uv - elementPos) > 0.0 &&
                               dot(elementPos - otherPos, uv - otherPos) > 0.0) {
                                color = mix(color, elementColor, 0.5 * strength);
                            }
                        }
                    }
                }
                
                // Draw element
                color = mix(color, elementColor, smoothstep(elementSize, 0.0, elementDist));
            }
        }
    }
    
    // Add subtle background gradient
    vec3 bgColor = vec3(0.02, 0.02, 0.05);
    color = mix(bgColor, color, smoothstep(1.0, 0.0, dist));
    
    outputColor = vec4(color, 1.0);
} 
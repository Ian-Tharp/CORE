// atmosphere.glsl.ts — BackSide fresnel shell + additive cloud shell. r165-correct.

export const atmosphereVert = /* glsl */`
  varying vec3 vWorldNormal;
  varying vec3 vWorldPos;
  void main() {
    vec4 wp = modelMatrix * vec4(position, 1.0);
    vWorldPos = wp.xyz;
    vWorldNormal = normalize(mat3(modelMatrix) * normal);
    gl_Position = projectionMatrix * viewMatrix * wp;
  }`;

export const atmosphereFrag = /* glsl */`
  precision highp float;
  uniform vec3  uColorInner;   // horizon glow
  uniform vec3  uColorOuter;   // limb / space edge
  uniform vec3  uSunDir;       // normalized, world space
  uniform float uIntensity;
  uniform float uPower;        // fresnel exponent (3..6)
  varying vec3  vWorldNormal;
  varying vec3  vWorldPos;
  void main() {
    // cameraPosition: r165 built-in uniform, injected for BOTH ortho and perspective.
    vec3 viewDir = normalize(cameraPosition - vWorldPos);
    float fres = pow(1.0 - max(dot(viewDir, -vWorldNormal), 0.0), uPower);
    float sun  = max(dot(-vWorldNormal, uSunDir), 0.0);
    float scatter = fres * (0.35 + 0.65 * sun);
    vec3 col = mix(uColorOuter, uColorInner, fres) * uIntensity;
    gl_FragColor = vec4(col, scatter); // AdditiveBlending consumes alpha as energy
    #include <colorspace_fragment>     // r165: encode to renderer output colour space
  }`;

// Clouds — density is baked per-vertex on the CPU (aCloud attribute) so no GLSL
// noise is needed; the fragment just emits additive white scaled by density.
export const cloudVert = /* glsl */`
  attribute float aCloud;
  varying float vCloud;
  void main() {
    vCloud = aCloud;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }`;

export const cloudFrag = /* glsl */`
  precision highp float;
  uniform vec3  uColor;
  uniform float uIntensity;
  varying float vCloud;
  void main() {
    float a = clamp(vCloud, 0.0, 1.0) * uIntensity;
    gl_FragColor = vec4(uColor * a, a);
    #include <colorspace_fragment>
  }`;

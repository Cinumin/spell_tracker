import moderngl_window as mglw

FIREBALL_FRAGMENT_SHADER = """
                #version 330
                uniform vec3 iResolution;
                uniform float iTime;
                uniform vec4 iMouse;
                out vec4 fragColor;

                float noise( vec3 P )
                {
                    vec3 Pi = floor(P);
                    vec3 Pf = P - Pi;
                    vec3 Pf_min1 = Pf - 1.0;
                    Pi.xyz = Pi.xyz - floor(Pi.xyz * ( 1.0 / 69.0 )) * 69.0;
                    vec3 Pi_inc1 = step( Pi, vec3( 69.0 - 1.5 ) ) * ( Pi + 1.0 );
                    vec4 Pt = vec4( Pi.xy, Pi_inc1.xy ) + vec2( 50.0, 161.0 ).xyxy;
                    Pt *= Pt;
                    Pt = Pt.xzxz * Pt.yyww;
                    vec2 hash_mod = vec2( 1.0 / ( 635.298681 + vec2( Pi.z, Pi_inc1.z ) * 48.500388 ) );
                    vec4 hash_lowz = fract( Pt * hash_mod.xxxx );
                    vec4 hash_highz = fract( Pt * hash_mod.yyyy );
                    vec3 blend = Pf * Pf * Pf * (Pf * (Pf * 6.0 - 15.0) + 10.0);
                    vec4 res0 = mix( hash_lowz, hash_highz, blend.z );
                    vec4 blend2 = vec4( blend.xy, vec2( 1.0 - blend.xy ) );
                    return dot( res0, blend2.zxzx * blend2.wwyy );
                }

                float fnoise(vec3 p, float time)
                {
                    float f = 0.0;
                    p = p - vec3(0.0, 1.0, 0.0) * .5 * time;
                    p = p * 3.0;
                    f += 0.50000 * noise(p); p = 2.0 * p;
                    f += 0.25000 * noise(p); p = 2.0 * p;
                    f += 0.12500 * noise(p); p = 2.0 * p;
                    f += 0.06250 * noise(p); p = 2.0 * p;
                    f += 0.03125 * noise(p);
                    return f;
                }

                float model( in vec3 p )
                {
                    p.y *= .7;
                    float sphere = length(p)-.8;
                    float res = sphere + fnoise(p*1.5, iTime*3.)* .4;
                    return res*.8;
                }

                float raymarch(in vec3 ro, in vec3 rd)
                {
                    float dist = 0.;
                    for(int i = 0; i < 30; i++)
                    {
                        float m = model(ro+rd*dist);
                        dist += m;
                        
                        if(m < .01) return dist;
                        else if(dist > 2.) break;
                    }
                    return -1.;
                }

                vec3 hueShift( vec3 color, float hueAdjust ){
                    const vec3 kRGBToYPrime = vec3 (0.299, 0.587, 0.114);
                    const vec3 kRGBToI      = vec3 (0.596, -0.275, -0.321);
                    const vec3 kRGBToQ      = vec3 (0.212, -0.523, 0.311);

                    const vec3 kYIQToR     = vec3 (1.0, 0.956, 0.621);
                    const vec3 kYIQToG     = vec3 (1.0, -0.272, -0.647);
                    const vec3 kYIQToB     = vec3 (1.0, -1.107, 1.704);

                    float YPrime  = dot (color, kRGBToYPrime);
                    float I       = dot (color, kRGBToI);
                    float Q       = dot (color, kRGBToQ);
                    float hue     = atan (Q, I);
                    float chroma  = sqrt (I * I + Q * Q);

                    hue += hueAdjust;

                    Q = chroma * sin (hue);
                    I = chroma * cos (hue);

                    vec3 yIQ   = vec3 (YPrime, I, Q);

                    return vec3( dot (yIQ, kYIQToR), dot (yIQ, kYIQToG), dot (yIQ, kYIQToB) );
                }

                vec3 saturation(vec3 rgb, float adjustment)
                {
                    const vec3 W = vec3(0.2125, 0.7154, 0.0721);
                    vec3 intensity = vec3(dot(rgb, W));
                    return mix(intensity, rgb, adjustment);
                }

                vec3 background(in vec2 p)
                {
                    return vec3(0.);
                }

                vec3 volume(in vec3 p, in vec3 rd, in vec3 ld, in vec2 sp)
                {
                    vec3 op = p;
                    float trans = 1.0;
                    float td = 0.0;
                    float dif = 0.0;
                    float emit = 0.0;
                    float steps = 30.; 
                    
                    for(float i = 0.; i < steps; i++)
                    {
                        float m = model(p);
                        p += rd*.03;
                        
                        float dens = 1.-smoothstep(0., .35, -m);
                        td += dens;
                        trans *= dens;
                        
                        if(td > 1.0 && dif <= 0.)
                        {
                            td = 1.;
                            dif = clamp(1. - model(p - .1*ld), 0., 1.);
                            emit = pow(smoothstep(-.3, 0., model(p)), 4.);
                        }
                    }
                    
                    trans = (1.-pow(td/steps, 4.5));
                    trans = smoothstep(0., 1., trans);
                    float emitmod = (emit*trans)*.8+.2;
                    
                    vec3 lin = vec3(.3, .2, .9);
                    lin = hueShift(lin, 4.2 + -trans*.6 + dif*.5);
                    lin *= emitmod;
                    
                    lin = saturation(lin, pow(trans, .5)*.4);
                    lin *= 5.;
                    lin -= vec3(.4);
                    
                    return mix(background(sp), lin, pow(trans, 1.25));
                }

                void mainImage( out vec4 fragColor, in vec2 fragCoord )
                {
                    vec2 p = (fragCoord - .5*iResolution.xy)/iResolution.y;
                    
                    float rs = .5;
                    vec3 ro = vec3(cos(iMouse.x/100.), 1., sin(iMouse.x/100.))*1.35;
                    vec3 ta = vec3(0., 0.1, .0);                
                    
                    vec3 w = normalize (ta-ro);
                    vec3 u = normalize (cross (w, vec3(0., 1., 0.)));
                    vec3 v = normalize (cross (u, w));
                    mat3 mat = mat3(u, v, w);
                    vec3 rd = normalize (mat*vec3(p.xy,1.));
                    
                    float dist = raymarch(ro, rd);
                    vec3 ld = vec3(-1., 1., 0.);
                    vec3 col = dist > 0. ? volume(ro+rd*dist, rd, ld, p) : background(p);
                    
                    fragColor = vec4(col,1.0);
                }

                void main() {
                    mainImage(fragColor, gl_FragCoord.xy);
                }
            """


class ShaderWindow(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "GLSL Raymarching"
    window_size = (800, 600)
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Full-screen quad geometry
        self.quad = mglw.geometry.quad_fs()

        # Compile Shader
        self.program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_position;
                void main() {
                    gl_Position = vec4(in_position, 0.0, 1.0);
                }
            """,
            fragment_shader=FIREBALL_FRAGMENT_SHADER,
        )

    def on_render(self, time: float, frametime: float):
        self.ctx.clear(0.0, 0.0, 0.0)

        # Update uniforms automatically
        if "iResolution" in self.program:
            self.program["iResolution"].value = (
                float(self.window_size[0]),
                float(self.window_size[1]),
                1.0,
            )
        if "iTime" in self.program:
            self.program["iTime"].value = time

        self.quad.render(self.program)

    def run_shader():
        mglw.run_window_config(ShaderWindow)

if __name__ == "__main__":
    ShaderWindow.run_shader()

//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kaszala Kristof
// Neptun : S9XEU5
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const int tessellationLevel = 100;

//---------------------------
struct Camera { // 3D camera
//---------------------------
	vec3 wEye, wLookat, wVup;   // extinsic
	float fov, asp, fp, bp;		// intrinsic
public:
	Camera() {
		asp = (float)windowWidth / windowHeight;
		fov = 70.0f * (float)M_PI / 180.0f;
		fp = 0.1; bp = 200;
	}
	mat4 V() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
			u.y, v.y, w.y, 0,
			u.z, v.z, w.z, 0,
			0, 0, 0, 1);
	}

	mat4 P() { // projection matrix
		return mat4(1 / (tan(fov / 2) * asp), 0, 0, 0,
			0, 1 / tan(fov / 2), 0, 0,
			0, 0, -(fp + bp) / (bp - fp), -1,
			0, 0, -2 * fp * bp / (bp - fp), 0);
	}

	mat4 Vinv() { // view matrix: translates the center to the origin
		vec3 w = normalize(wEye - wLookat);
		vec3 u = normalize(cross(wVup, w));
		vec3 v = cross(w, u);
		return mat4(u.x, u.y, u.z, 0,
			v.x, v.y, v.z, 0,
			w.x, w.y, w.z, 0,
			0, 0, 0, 1) * TranslateMatrix(wEye);
	}
	mat4 Pinv() { // view matrix: translates the center to the origin
		float scale = tan(fov / 2) * length(wEye - wLookat);
		return mat4(scale, 0, 0, 0,
			0, scale, 0, 0,
			0, 0, -(bp - fp) / 2, 0,
			0, 0, -(fp + bp) / 2, 1);
	}

	void Animate(float t);
};
Camera camera;

//---------------------------
struct Material {
	//---------------------------
	vec3 kd, ks, ka;
	float shininess;
};

vec4 qmul(vec4 q1, vec4 q2) {
	vec3 d1(q1.x, q1.y, q1.z); vec3 d2(q2.x, q2.y, q2.z);
	vec3 tmp = d2 * q1.w + d1 * q2.w + cross(d1, d2);
	return vec4(tmp.x, tmp.y, tmp.z, q1.w * q2.w - dot(d1, d2));
}

vec3 Rotate(vec3 u, vec4 q) {
	vec4 qinv(-q.x, -q.y, -q.z, q.w);
	vec4 qr = qmul(qmul(q, vec4(u.x, u.y, u.z, 0)), qinv);
	return vec3(qr.x, qr.y, qr.z);
}

vec3 vec4Tovec3(const vec4& v) {
	return vec3(v.x, v.y, v.z);
}
vec4 vec3Tovec4(const vec3& v, float f = 0) {
	return vec4(v.x, v.y, v.z, f);
}
//---------------------------
struct Light {
	//---------------------------
	vec3 La, Le;
	vec4 wLightPos;
	vec4 initPos;

	void Animate(float t) {
		vec4 q = vec4(cosf(t / 4.0f),
			sinf(t / 4.0f) * cosf(t) / 2.0f,
			sinf(t / 4.0f) * sinf(t) / 2.0f,
			sinf(t / 4.0f) * sqrtf(0.75f));

		wLightPos = vec3Tovec4(Rotate(vec4Tovec3(initPos), q), 0);
	}
};

//---------------------------
class CheckerBoardTexture : public Texture {
	//---------------------------
public:
	CheckerBoardTexture(const int width = 0, const int height = 0) : Texture() {
		std::vector<vec4> image(width * height);
		const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
		}
		create(width, height, image, GL_NEAREST);
	}
};

//---------------------------
struct RenderState {
	//---------------------------
	mat4	           MVP, M, Minv, V, P;
	Material* material;
	std::vector<Light> lights;
	Texture* texture;
	vec3	           wEye;
};

//---------------------------
class Shader : public GPUProgram {
	//---------------------------
public:
	virtual void Bind(RenderState state) = 0;

	void setUniformMaterial(const Material& material, const std::string& name) {
		setUniform(material.kd, name + ".kd");
		setUniform(material.ks, name + ".ks");
		setUniform(material.ka, name + ".ka");
		setUniform(material.shininess, name + ".shininess");
	}

	void setUniformLight(const Light& light, const std::string& name) {
		setUniform(light.La, name + ".La");
		setUniform(light.Le, name + ".Le");
		setUniform(light.wLightPos, name + ".wLightPos");
	}
};

//---------------------------
class GouraudShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};
		
		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform mat4  MVP, M, Minv;  // MVP, Model, Model-inverse
		uniform Light[8] lights;     // light source direction 
		uniform int   nLights;		 // number of light sources
		uniform vec3  wEye;          // pos of eye
		uniform Material  material;  // diffuse, specular, ambient ref

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space

		out vec3 radiance;		    // reflected radiance

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;	
			vec3 V = normalize(wEye * wPos.w - wPos.xyz);
			vec3 N = normalize((Minv * vec4(vtxNorm, 0)).xyz);
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein

			radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				radiance += material.ka * lights[i].La + (material.kd * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		in  vec3 radiance;      // interpolated radiance
		out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	GouraudShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
class PhongShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal;		    // normal in world space
		out vec3 wView;             // view in world space
		out vec3 wLight[8];		    // light dir in world space
		out vec2 texcoord;

		void main() {
			gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
			// vectors for radiance computation
			vec4 wPos = vec4(vtxPos, 1) * M;
			for(int i = 0; i < nLights; i++) {
				wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
			}
		    wView  = wEye * wPos.w - wPos.xyz;
		    wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		    texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		struct Light {
			vec3 La, Le;
			vec4 wLightPos;
		};

		struct Material {
			vec3 kd, ks, ka;
			float shininess;
		};

		uniform Material material;
		uniform Light[8] lights;    // light sources 
		uniform int   nLights;
		uniform sampler2D diffuseTexture;

		in  vec3 wNormal;       // interpolated world sp normal
		in  vec3 wView;         // interpolated world sp view
		in  vec3 wLight[8];     // interpolated world sp illum dir
		in  vec2 texcoord;
		
        out vec4 fragmentColor; // output goes to frame buffer

		void main() {
			vec3 N = normalize(wNormal);
			vec3 V = normalize(wView); 
			if (dot(N, V) < 0) N = -N;	// prepare for one-sided surfaces like Mobius or Klein
			vec3 texColor = texture(diffuseTexture, texcoord).rgb;
			vec3 ka = material.ka * texColor;
			vec3 kd = material.kd * texColor;

			vec3 radiance = vec3(0, 0, 0);
			for(int i = 0; i < nLights; i++) {
				vec3 L = normalize(wLight[i]);
				vec3 H = normalize(L + V);
				float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
				// kd and ka are modulated by the texture
				radiance += ka * lights[i].La + 
                           (kd * texColor * cost + material.ks * pow(cosd, material.shininess)) * lights[i].Le;
			}
			fragmentColor = vec4(radiance, 1);
		}
	)";
public:
	PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniformMaterial(*state.material, "material");

		setUniform((int)state.lights.size(), "nLights");
		for (unsigned int i = 0; i < state.lights.size(); i++) {
			setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
		}
	}
};

//---------------------------
class NPRShader : public Shader {
	//---------------------------
	const char* vertexSource = R"(
		#version 330
		precision highp float;

		uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
		uniform	vec4  wLightPos;
		uniform vec3  wEye;         // pos of eye

		layout(location = 0) in vec3  vtxPos;            // pos in modeling space
		layout(location = 1) in vec3  vtxNorm;      	 // normal in modeling space
		layout(location = 2) in vec2  vtxUV;

		out vec3 wNormal, wView, wLight;				// in world space
		out vec2 texcoord;

		void main() {
		   gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
		   vec4 wPos = vec4(vtxPos, 1) * M;
		   wLight = wLightPos.xyz * wPos.w - wPos.xyz * wLightPos.w;
		   wView  = wEye * wPos.w - wPos.xyz;
		   wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
		   texcoord = vtxUV;
		}
	)";

	// fragment shader in GLSL
	const char* fragmentSource = R"(
		#version 330
		precision highp float;

		uniform sampler2D diffuseTexture;

		in  vec3 wNormal, wView, wLight;	// interpolated
		in  vec2 texcoord;
		out vec4 fragmentColor;    			// output goes to frame buffer

		void main() {
		   vec3 N = normalize(wNormal), V = normalize(wView), L = normalize(wLight);
		   float y = (dot(N, L) > 0.5) ? 1 : 0.5;
		   if (abs(dot(N, V)) < 0.2) fragmentColor = vec4(0, 0, 0, 1);
		   else						 fragmentColor = vec4(y * texture(diffuseTexture, texcoord).rgb, 1);
		}
	)";
public:
	NPRShader() { create(vertexSource, fragmentSource, "fragmentColor"); }

	void Bind(RenderState state) {
		Use(); 		// make this program run
		setUniform(state.MVP, "MVP");
		setUniform(state.M, "M");
		setUniform(state.Minv, "Minv");
		setUniform(state.wEye, "wEye");
		setUniform(*state.texture, std::string("diffuseTexture"));
		setUniform(state.lights[0].wLightPos, "wLightPos");
	}
};

//---------------------------
struct VertexData {
	//---------------------------
	vec3 position, normal;
	vec2 texcoord;
};

//---------------------------
class Geometry {
	//---------------------------
protected:
	unsigned int vao, vbo;        // vertex array object
public:
	Geometry() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() = 0;
	virtual vec3 getXYZ(float u, float v) { return 0; }
	virtual float getZ(float u, float v) { return 0; }
	~Geometry() {
		glDeleteBuffers(1, &vbo);
		glDeleteVertexArrays(1, &vao);
	}
};

//---------------------------
class ParamSurface : public Geometry {
	//---------------------------
public:
	unsigned int nVtxPerStrip, nStrips;
	ParamSurface() { nVtxPerStrip = nStrips = 0; }

	virtual VertexData GenVertexData(float u, float v) = 0;


	void create(int N = tessellationLevel, int M = tessellationLevel) {
		nVtxPerStrip = (M + 1) * 2;
		nStrips = N;
		std::vector<VertexData> vtxData;	// vertices on the CPU
		for (int i = 0; i < N; i++) {
			for (int j = 0; j <= M; j++) {
				vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
				vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
			}
		}
		glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), &vtxData[0], GL_STATIC_DRAW);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
		glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
		glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
		// attribute array, components/attribute, component type, normalize?, stride, offset
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, texcoord));
	}

	void Draw() {
		glBindVertexArray(vao);
		for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i * nVtxPerStrip, nVtxPerStrip);
	}
};

class Sphere : public ParamSurface {
	//---------------------------
	VertexData vd;
public:
	Sphere() { create(); }

	VertexData GenVertexData(float u, float v) {
		vd.position = vd.normal = vec3(cosf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
			sinf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
			cosf(v * (float)M_PI));
		vd.texcoord = vec2(u, v);
		return vd;
	}

	float getZ(float u, float v) override {
		return vec3(cosf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
			sinf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
			cosf(v * (float)M_PI)).z;
	}

	vec3 getXYZ(float u, float v) { 
		return vec3(cosf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
			sinf(u * 2.0f * (float)M_PI) * sinf(v * (float)M_PI),
			cosf(v * (float)M_PI));
	}
};

class Surface : public ParamSurface {
public:
	float W, H, D, K, phase1, phase2;
	VertexData vd;
	Surface() { W = 60.0f; H = 60.0f; K = 8; D = 2; phase1 = 1; phase2 = 2.8; create(); }

	VertexData GenVertexData(float u, float v) {
		float angle = u * K * M_PI +phase1;
		float angle2 = v * K * M_PI +phase2;

		vd.position = vec3(u * W, v * H, (sinf(angle) + cosf(angle2)) * D);
		vd.normal = vec3(K * D * M_PI * (sinf(angle) - cosf(angle)), 0, W);
		vd.normal = normalize(vd.normal);
		vd.texcoord = vec2(u, v);
		return vd;
	}

	float getZ(float u, float v) override {
		float angle = u * K * M_PI + phase1;
		float angle2 = v * K * M_PI + phase2;
		return vec3(u * W, v * H, (sinf(angle) + cosf(angle2)) * D).z;
	}
	vec3 getXYZ(float u, float v) override {
		float angle = u * K * M_PI + phase1;
		float angle2 = v * K * M_PI + phase2;
		return vec3(u * W, v * H, (sinf(angle) + cosf(angle2)) * D);
	}
};

//---------------------------
struct Object {
	//---------------------------
	Shader* shader;
	Material* material;
	Texture* texture;
	Geometry* geometry;
	vec3 scale, translation, rotationAxis;
	float rotationAngle;
public:
	Object(Shader* _shader, Material* _material, Texture* _texture, Geometry* _geometry) :
		scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
		shader = _shader;
		texture = _texture;
		material = _material;
		geometry = _geometry;
	}
	virtual void SetModelingTransform(mat4& M, mat4& Minv) {
		M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
		Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
	}

	void Draw(RenderState state) {
		mat4 M, Minv;
		SetModelingTransform(M, Minv);
		state.M = M;
		state.Minv = Minv;
		state.MVP = state.M * state.V * state.P;
		state.material = material;
		state.texture = texture;
		shader->Bind(state);
		geometry->Draw();
	}

	virtual void Animate(float tstart, float tend) { rotationAngle = 0.8f * tend; }
};

class CheckerBoardTexture2 : public Texture {
	//---------------------------
public:
	CheckerBoardTexture2(const Object* surface, const int width = 0, const int height = 0) {
		std::vector<vec4> image(width * height);


		float zmin = 999999;
		float zmax = -999999;
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			float z = surface->geometry->getZ((float)x / width, (float)y / height);
			if (z < zmin) { zmin = z; }
			if (z > zmax) { zmax = z; }
		}

		const vec4 green(0, 1, 0, 1), brown(0.6, 0.15, 0.3, 1);
		for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
			float z = surface->geometry->getZ((float)x / (float)width, (float)y / (float)height);
			vec4 newColor(0, 0, 0, 1);
			newColor.x = green.x + ((z - zmin) * (brown.x - green.x)) / (zmax - zmin);
			newColor.y = green.y + ((z - zmin) * (brown.y - green.y)) / (zmax - zmin);
			newColor.z = green.z + ((z - zmin) * (brown.z - green.z)) / (zmax - zmin);
			image[y * width + x] = newColor;
		}
		create(width, height, image, GL_NEAREST);
	}
};

//GPUProgram gpuProgram;
const int nTesselatedVertices = 1000;

class Curve : public GPUProgram {
public:
	unsigned int vaoCurve, vboCurve;
	unsigned int vaoCtrlPoints, vboCtrlPoints;

	std::vector<vec4> wCtrlPoints;
	std::vector<vec4> points1000;

	Curve() {
		glGenVertexArrays(1, &vaoCurve);
		glBindVertexArray(vaoCurve);

		glGenBuffers(1, &vboCurve);
		glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL);

		glGenVertexArrays(1, &vaoCtrlPoints);
		glBindVertexArray(vaoCtrlPoints);

		glGenBuffers(1, &vboCtrlPoints);
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
	}

	~Curve() {
		glDeleteBuffers(1, &vboCtrlPoints); glDeleteVertexArrays(1, &vaoCtrlPoints);
		glDeleteBuffers(1, &vboCurve); glDeleteVertexArrays(1, &vaoCurve);
	}

	virtual vec4 r(float t) { return wCtrlPoints[0]; }

	virtual void AddControlPoint(vec2 cPoint) {
		vec4 wVertex = vec4(cPoint.x, cPoint.y, 0, 1);
		wCtrlPoints.push_back(wVertex);
	}

	vec2 to2(vec4 a) {
		return vec2(a.x, a.y);
	}

	void Draw() {
		mat4 ct = camera.V() * camera.P();

		if (wCtrlPoints.size() >= 2) {
			std::vector<float> vertexData;
			for (int i = 0; i < 1000; i++) {
				vec4 wVertex = r((float)i / (float)1000);
				vertexData.push_back(wVertex.x);
				vertexData.push_back(wVertex.y);
			}
			glBindVertexArray(vaoCurve);
			glBindBuffer(GL_ARRAY_BUFFER, vboCurve);
			glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
			glDrawArrays(GL_LINE_STRIP, 0, nTesselatedVertices);
		}
	}
};

class CatmullRom : public Curve {
public:
	std::vector<float> t;
	float dt;

	vec4 Hermite(vec4 p0, vec4 v0, float t0, vec4 p1, vec4 v1, float t1, float t) {
		vec4 a0 = p0;
		vec4 a1 = v0;
		vec4 a2 = (((p1 - p0) * 3) / ((t1 - t0) * (t1 - t0))) - ((v1 + v0 * 2) / (t1 - t0));
		vec4 a3 = (((p0 - p1) * 2) / ((t1 - t0) * (t1 - t0) * (t1 - t0))) + ((v1 + v0) / ((t1 - t0) * (t1 - t0)));
		vec4 ret = a3 * (t - t0) * (t - t0) * (t - t0) + a2 * (t - t0) * (t - t0) + a1 * (t - t0) + a0;
		return ret;
	}

	CatmullRom() {}

	void AddControlPoint(vec2 x) {
		vec4 wVertex = vec4(x.x, x.y, 0.0f, 1) * camera.Pinv() * camera.Vinv();
		if (wCtrlPoints.size() >= 3) {
			wCtrlPoints.pop_back();
		}
		wCtrlPoints.push_back(wVertex) ;
		if (wCtrlPoints.size() >= 3) {
			wCtrlPoints.push_back(wCtrlPoints[0]);
		}
		this->t.clear();
		for (int i = 0; i < wCtrlPoints.size(); i++)
			t.push_back((float)i / (float)(wCtrlPoints.size() - 1));
	}

	vec4 r(float t) {
		for (int i = 0; i < wCtrlPoints.size() - 1; i++) {
			if (this->t[i] <= t && t <= this->t[i + 1]) {
				vec4 v0, v1;
				if (i == 0) { v0.x = v0.y = v0.z = 0.0f; v0.w = 1.0f; }
				else {
					v0 = (wCtrlPoints[i + 1] - wCtrlPoints[i]) / (this->t[i + 1] - this->t[i]) + (wCtrlPoints[i] - wCtrlPoints[i - 1]) / (this->t[i] - this->t[i - 1]);
				}

				if (i == wCtrlPoints.size() - 2) { v1.x = v1.y = v1.z = 0.0f; v0.w = 1.0f; }
				else {
					v1 = (wCtrlPoints[i + 2] - wCtrlPoints[i + 1]) / (this->t[i + 2] - this->t[i + 1]) + (wCtrlPoints[i + 1] - wCtrlPoints[i]) / (this->t[i + 1] - this->t[i]);
				}

				return Hermite(wCtrlPoints[i], v0, this->t[i], wCtrlPoints[i + 1], v1, this->t[i + 1], t);
			}
		}
	}
};
Curve* curve;

//vec2 toNDC(int pX, int pY) {
//	vec2 cPoint;
//	cPoint.x = (2.44f * pX / windowWidth - 0.4f) * 30.0f;
//	cPoint.y = (2.0f* pY / windowHeight) * 10.0f;
//	return cPoint;
//}

vec2 toNDC(int pX, int pY) {
	vec2 cPoint;
	cPoint.x = 2.0f * pX / windowWidth - 1;
	cPoint.y = 1.0f - 2.0f * pY / windowHeight;
	return cPoint;
}

//---------------------------
class Scene {
public:
	//---------------------------
	std::vector<Object*> objects;
	std::vector<Light> lights;
	void Build() {
		// Shaders
		Shader* phongShader = new PhongShader();
		Shader* gouraudShader = new GouraudShader();
		Shader* nprShader = new NPRShader();

		// Materials
		Material* material1 = new Material;
		material1->kd = vec3(0.9f, 0.7f, 0.3f);
		material1->ks = vec3(0.3f, 0.3f, 0.3f);   // 1,1,1 too shiny
		material1->ka = vec3(0.3f, 0.3f, 0.3f);
		material1->shininess = 100;

		// Textures
		Texture* texture4x8 = new CheckerBoardTexture(4, 8);
		Texture* texture15x20 = new CheckerBoardTexture(15, 20);

		// Geometries
		Geometry* sphere = new Sphere();
		Geometry* surface = new Surface();

		Object* surfaceTest = new Object(phongShader, material1, NULL, surface);
		Texture* texture512x512 = new CheckerBoardTexture2(surfaceTest, 512, 512);
		surfaceTest->texture = texture512x512;
		surfaceTest->translation = vec3(-27, -30, 0);
		surfaceTest->rotationAxis = vec3(1, 1, 0);
		//surfaceTest->scale = vec3(0.5f, 1.2f, 0.5f);
		objects.push_back(surfaceTest);

		// Camera
		//camera.wEye = vec3(13, 10, 30);  
		//camera.wLookat = vec3(13, 10, 0);
		//camera.wVup = vec3(0, 1, 0);

		camera.wEye = vec3(0, 0, 20);
		camera.wLookat = vec3(0, 0, 0);
		camera.wVup = vec3(0, 1, 0);

		// Lights
		lights.resize(2);

		lights[0].wLightPos = vec4(0, 0, 10, 0);	// ideal point -> directional light source
		lights[0].La = vec3(0.3f, 0.3f, 0.3f);
		lights[0].Le = vec3(1.8, 2.4,1);

		lights[1].wLightPos = vec4(10, 10, 0, 0);
		lights[1].La = vec3(0.3f, 0.3f, 0.3f);
		lights[1].Le = vec3(1.8, 2.4, 1);

		lights[0].initPos = lights[1].wLightPos;
		lights[1].initPos = lights[0].wLightPos;
	}

	void Render() {
		RenderState state;
		state.wEye = camera.wEye;
		state.V = camera.V();
		state.P = camera.P();
		state.lights = lights;
		for (Object* obj : objects) obj->Draw(state);
	}

	void Animate(float tstart, float tend) {
		for (unsigned int i = 0; i < lights.size(); i++) { lights[i].Animate(tend); }
		//for (Object* obj : objects) obj->Animate(tstart, tend);
	}
};
static bool pressed = false;
Scene scene;

void Camera::Animate(float t) {
	wEye = vec3(-6.0f+1.0f*cos(t/10.0f)*cos(t/10.0f),     -55.0f+t,     3.0f+sin(t)+cos(t));
		wLookat = vec3(-6.0f, 300.0f, 0);
}

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	scene.Build();
	curve = new CatmullRom();
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0.5f, 0.5f, 0.8f, 1.0f);							// background color 
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
	scene.Render();
	if(!pressed)
		curve->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { 
		pressed = true;
		if (key == 'w') {
			Material* material1 = new Material;
			material1->kd = vec3(0.9f, 0.7f, 0.3f);
			material1->ks = vec3(0.3f, 0.3f, 0.3f);   // 1,1,1 too shiny
			material1->ka = vec3(0.3f, 0.3f, 0.3f);
			material1->shininess = 10;
			Texture* texture15x20 = new CheckerBoardTexture(15, 20);
			Geometry* sphere = new Sphere();
			float j = 0;
			for (int i = 0; i < 1000; i++) {
				vec4 wVertex = curve->r((float)i / 1000.0f);
				curve->points1000.push_back(wVertex);

				Object* tmp = new Object(new PhongShader(), material1, texture15x20, sphere);
				float fX, fY, fZ;
				fX = curve->points1000[i].x;
				fY = curve->points1000[i].y;

				float sphereZ = scene.objects[0]->geometry->getZ(curve->points1000[i].x / 60.0f, curve->points1000[i].y / 60.0f);

				tmp->translation = vec3(fX + 0.5f, fY + 1.0f, sphereZ + 0.1f);
				tmp->scale = vec3(0.1f, 0.1f, 0.1f);
				scene.objects.push_back(tmp);
			}
		}

		if (key == 'w') {
			camera.wEye.x += 1.0f;
		}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		curve->AddControlPoint(toNDC(pX, pY));
		glutPostRedisplay();
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	static float tend = 0;
	const float dt = 0.1f; // dt is ”infinitesimal”
	float tstart = tend;
	tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
	for (float t = tstart; t < tend; t += dt) {
		float Dt = fmin(dt, tend - t);
		scene.Animate(t, t + Dt);
		
			if(pressed)
				camera.Animate(t);
	}
	glutPostRedisplay();
}
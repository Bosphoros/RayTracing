// This code is highly based on smallpt
// http://www.kevinbeason.com/smallpt/
#include <cmath>
#include <algorithm>
#include <cassert>
#include <random>
#include <memory>
#include <fstream>
#include <iostream>

// GLM (vector / matrix)
#define GLM_FORCE_RADIANS

#include <glm/vec4.hpp>
#include <glm/vec3.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>

const float pi = 3.1415927f;
const float noIntersect = std::numeric_limits<float>::infinity();

bool isIntersect(float t)
{
    return t < noIntersect;
}

struct Ray
{
    const glm::vec3 origin, direction;
};

struct Sphere
{
    const float radius;
    const glm::vec3 center;
};

struct Triangle
{
    const glm::vec3 v0, v1, v2;
};

    // WARRING: works only if r.d is normalized
float intersect (const Ray & ray, const Sphere &sphere)
{				// returns distance, 0 if nohit
    glm::vec3 op = sphere.center - ray.origin;		// Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
    float t, b = glm:: dot(ray.direction, op), det =
        b * b - glm::dot(op, op) + sphere.radius * sphere.radius;
    if (det < 0)
        return noIntersect;
    else
        det = std::sqrt (det);
    return (t = b - det) >= 0 ? t : ((t = b + det) >= 0 ? t : noIntersect);
}

float intersect(const Ray & ray, const Triangle &triangle)
{
    auto e1 = triangle.v1 - triangle.v0;
    auto e2 = triangle.v2 - triangle.v0;

    auto h = glm::cross(ray.direction, e2);
    auto a = glm::dot(e1, h);

    auto f = 1.f / a;
    auto s = ray.origin - triangle.v0;

    auto u = f * glm::dot(s, h);
    auto q = glm::cross(s, e1);
    auto v = f * glm::dot(ray.direction, q);
    auto t = f * glm::dot(e2, q);

    if(std::abs(a) < 0.00001)
        return noIntersect;
    if(u < 0 || u > 1)
        return noIntersect;
    if(v < 0 || (u+v) > 1)
        return noIntersect;
    if(t < 0)
        return noIntersect;

    return t;
}

glm::vec3 getNormale(const glm::vec3 &point, const Sphere &sphere) {
    return glm::normalize(point-sphere.center);
}

glm::vec3 getNormale(const glm::vec3 &point, const Triangle &triangle) {
    return glm::normalize(glm::cross(glm::vec3{triangle.v1-triangle.v0},glm::vec3{triangle.v2-triangle.v0}));
}

struct Diffuse
{
    const glm::vec3 color;
};

struct Glass
{
    const glm::vec3 color;
};

struct Mirror
{
    const glm::vec3 color;
};

glm::vec3 indirect(const Ray &rOrigine, const Ray &rReflect, const glm::vec3 & n, int countdown, const Diffuse &diffuse);
glm::vec3 indirect(const Ray &rOrigine, const Ray &rReflect, const glm::vec3 & n, int countdown, const Glass &glass);
glm::vec3 indirect(const Ray &rOrigine, const Ray &rReflect, const glm::vec3 & n, int countdown, const Mirror &mirror);

double bsdf(const double a, const Diffuse &diffuse) {
    return a/pi;
}

double bsdf(const double a, const Glass &glass) {
    return 0;
}

double bsdf(const double a, const Mirror &mirror) {
    return 0;
}

template<typename T>
glm::vec3 albedo(const T &t)
{
    return t.color;
}

struct Object
{
    virtual float intersect(const Ray &r) const = 0;
    virtual glm::vec3 albedo() const = 0;
    virtual glm::vec3 getNormale(const glm::vec3 &point) const = 0;
    virtual float bsdf(const double a) const = 0;
    virtual glm::vec3 indirect(const Ray &rOrigine, const Ray &rReflect, const glm::vec3 & n, int countdown) const = 0;
};

template<typename P, typename M>
struct ObjectTpl final : Object
{
    ObjectTpl(const P &_p, const M &_m)
        :primitive(_p), material(_m)
    {}

    float intersect(const Ray &ray) const
    {
        return ::intersect(ray, primitive);
    }

    glm::vec3 albedo() const
    {
        return ::albedo(material);
    }

    glm::vec3 getNormale(const glm::vec3 &point) const
    {
        return ::getNormale(point, primitive);
    }

    float bsdf(const double a) const
    {
        return ::bsdf(a, material);
    }

    glm::vec3 indirect(const Ray &rOrigine, const Ray &rReflect, const glm::vec3 & n, int countdown) const
    {
        return albedo()*::indirect(rOrigine, rReflect, n, countdown, material);
    }

    const P &primitive;
    const M &material;
};


template<typename P, typename M>
std::unique_ptr<Object> makeObject(const P&p, const M&m)
{
    return std::unique_ptr<Object>(new ObjectTpl<P, M>{p, m});
}

// Scene
namespace scene
{
    // Primitives

    // Left Wall
    const Triangle leftWallA{{0, 0, 0}, {0, 0, 150}, {0, 100, 0}};
    const Triangle leftWallB{{0, 100, 150}, {0, 100, 0}, {0, 0, 150}};

    // Right Wall
    const Triangle rightWallA{{100, 0, 0}, {100, 100, 0}, {100, 0, 150}};
    const Triangle rightWallB{{100, 100, 150}, {100, 0, 150}, {100, 100, 0}};

    // Back wall
    const Triangle backWallA{{0, 0, 0}, {100, 0, 0}, {100, 100, 0}};
    const Triangle backWallB{{0, 0, 0}, {100, 100, 0}, {0, 100, 0}};

    // Bottom Floor
    const Triangle bottomWallA{{0, 0, 0}, {100, 0, 0}, {100, 0, 150}};
    const Triangle bottomWallB{{0, 0, 0}, {100, 0, 150}, {0, 0, 150}};

    // Top Ceiling
    const Triangle topWallA{{0, 100, 0}, {100, 100, 0}, {0, 100, 150}};
    const Triangle topWallB{{100, 100, 150}, {0, 100, 150}, {100, 100, 0}};

    const Sphere leftSphere{16.5, glm::vec3 {27, 16.5, 47}};
    const Sphere rightSphere{16.5, glm::vec3 {73, 16.5, 78}};

    const glm::vec3 light{50, 70, 81.6};
    const glm::vec3 lightColor(10,10,10);

    // Materials
    const Diffuse white{{.75, .75, .75}};
    const Diffuse red{{.75, .25, .25}};
    const Diffuse blue{{.25, .25, .75}};

    const Glass glass{{1, 1, 1}};
    const Mirror mirror{{1, 1, 1}};

    // Objects
    // Note: this is a rather convoluted way of initialising a vector of unique_ptr ;)
    const std::vector<std::unique_ptr<Object>> objects = [] (){
        std::vector<std::unique_ptr<Object>> ret;
        ret.push_back(makeObject(backWallA, white));
        ret.push_back(makeObject(backWallB, white));
        ret.push_back(makeObject(topWallA, white));
        ret.push_back(makeObject(topWallB, white));
        ret.push_back(makeObject(bottomWallA, white));
        ret.push_back(makeObject(bottomWallB, white));
        ret.push_back(makeObject(rightWallA, blue));
        ret.push_back(makeObject(rightWallB, blue));
        ret.push_back(makeObject(leftWallA, red));
        ret.push_back(makeObject(leftWallB, red));

        ret.push_back(makeObject(leftSphere, mirror));
        ret.push_back(makeObject(rightSphere, glass));

        return ret;
    }();
}

thread_local std::default_random_engine generator;
thread_local std::uniform_real_distribution<float> distribution(0.0,1.0);

float random_u()
{
    return distribution(generator);
}

glm::vec3 sample_cos(const float u, const float v, const glm::vec3 n)
{
    // Ugly: create an ornthogonal base
    glm::vec3 basex, basey, basez;

    basez = n;
    basey = glm::vec3(n.y, n.z, n.x);

    basex = glm::cross(basez, basey);
    basex = glm::normalize(basex);

    basey = glm::cross(basez, basex);

    // cosinus sampling. Pdf = cosinus
    return  basex * (std::cos(2.f * pi * u) * std::sqrt(1.f - v)) +
        basey * (std::sin(2.f * pi * u) * std::sqrt(1.f - v)) +
        basez * std::sqrt(v);
}

int toInt (const float x)
{
    return int (std::pow (glm::clamp (x, 0.f, 1.f), 1.f / 2.2f) * 255 + .5);
}

// WARNING: ASSUME NORMALIZED RAY
// Compute the intersection ray / scene.
// Returns true if intersection
// t is defined as the abscisce along the ray (i.e
//             p = r.o + t * r.d
// id is the id of the intersected object
Object* intersect (const Ray & r, float &t)
{
    t = noIntersect;
    Object *ret = nullptr;

    for(auto &object : scene::objects)
    {
        float d = object->intersect(r);
        if (isIntersect(d) && d < t)
        {
            t = d;
            ret = object.get();
        }
    }

    return ret;
}

// Reflect the ray i along the normal.
// i should be oriented as "leaving the surface"
glm::vec3 reflect(const glm::vec3 i, const glm::vec3 n)
{
    return n * (glm::dot(n, i)) * 2.f - i;
}

float sin2cos (const float x)
{
    return std::sqrt(std::max(0.0f, 1.0f-x*x));
}

// Fresnel coeficient of transmission.
// Normal point outside the surface
// ior is n0 / n1 where n0 is inside and n1 is outside
float fresnelR(const glm::vec3 i, const glm::vec3 n, const float ior)
{
    if(glm::dot(n, i) < 0)
        return fresnelR(i, n * -1.f, 1.f / ior);

    float R0 = (ior - 1.f) / (ior + 1.f);
    R0 *= R0;

    return R0 + (1.f - R0) * std::pow(1.f - glm::dot(i, n), 5.f);
}

// compute refraction vector.
// return true if refraction is possible.
// i and n are normalized
// output wo, the refracted vector (normalized)
// n point oitside the surface.
// ior is n00 / n1 where n0 is inside and n1 is outside
//
// i point outside of the surface
bool refract(glm::vec3 i, glm::vec3 n, float ior, glm::vec3 &wo)
{
    i = i * -1.f;

    if(glm::dot(n, i) > 0)
    {
        n = n * -1.f;
    }
    else
    {
        ior = 1.f / ior;
    }

    float k = 1.f - ior * ior * (1.f - glm::dot(n, i) * glm::dot(n, i));
    if (k < 0.)
        return false;

    wo = i * ior - n * (ior * glm::dot(n, i) + std::sqrt(k));

    return true;
}

/* Retourne un vecteur de déplacement
 * r rayon de la sphere
 * u, v randoms
 * pdf valeur out
 * normal normale (direction lumière/point normalisée)
*/
glm::vec3 sample_sphere(const float r, const float u, const float v, float &pdf, const glm::vec3 normal)
{
    pdf = 1.f / (pi * r * r);
    glm::vec3 sample_p = sample_cos(u, v, normal);

    float cos = glm::dot(sample_p, normal);

    pdf *= cos;
    return sample_p * r;
}

// dirLux normalisé
glm::vec3 random_light(const glm::vec3 dirLux, float &pdf) {
    float u = random_u();
    float v = random_u();
    return sample_sphere(10.f, u, v, pdf, -dirLux);
}

float squaredDistance(const glm::vec3 &v) {
    return v.x*v.x+v.y*v.y+v.z*v.z;
}

glm::vec3 radiance (const Ray & r, int countdown = 6)
{
    float t;
    Object* o = intersect(r, t);
    if(t == noIntersect)
        return glm::vec3{0,0,0};

    // Le point et sa normale
    glm::vec3 p = r.origin+t*r.direction;
    glm::vec3 normale = o->getNormale(p);

    // Choix d'un point aléatoire sur une sphère de lumière
    glm::vec3 lightPToSphere = scene::light-p;
    glm::vec3 dirToSphere = glm::normalize(lightPToSphere);
    float pdf;
    glm::vec3 lightOnSphere = random_light(dirToSphere, pdf);


    //std::cout << scene::light.x << "," << scene::light.y << "," << scene::light.z << " / " << lightOnSphere.x << "," << lightOnSphere.y << "," << lightOnSphere.z << std::endl;
    // Intersection entre le point et la lumière
    glm::vec3 lightP = (scene::light+lightOnSphere)-p;
    glm::vec3 dir = glm::normalize(lightP);
    glm::vec3 p2 = p+dir*0.1f;
    Ray ray{p2,dir};
    intersect(ray, t);

    // Rayon réfracté par la surface
    glm::vec3 ex = reflect(-r.direction, normale);
    glm::vec3 pEx = p + ex*0.1f;
    Ray rayEx{pEx, ex};
    float light = 1.0f;

    // Si intersection avant lumière alors pas d'éclairage direct
    if(t*t < squaredDistance(lightP))
        light = 0;

    double angle = glm::dot(normale,dir);

    //return o->albedo()*energie;

    if(countdown > 0)
        return o->albedo()*o->bsdf(std::abs(angle))*light*scene::lightColor/squaredDistance(lightP)/pdf + o->indirect(r, rayEx, normale, --countdown);
    else
        return o->albedo()*o->bsdf(std::abs(angle))*light*scene::lightColor/squaredDistance(lightP)/pdf;
}


glm::vec3 indirect(const Ray &rOrigine, const Ray &rReflect, const glm::vec3 & n, int countdown, const Diffuse &diffuse) {
    float pdf;
    glm::vec3 w = glm::normalize(sample_sphere(1, random_u(), random_u(), pdf, n));
    Ray rIndirect{rReflect.origin-0.1f*rReflect.direction+0.1f*w, w};
    return radiance(rIndirect, countdown);
    //return glm::vec3(0,0,0);
}

glm::vec3 indirect(const Ray &rOrigine, const Ray &rReflect, const glm::vec3 & n, int countdown, const Glass &glass) {

    float fresnel = fresnelR(-rOrigine.direction,n, 1.5);
    glm::vec3 refracted;
    bool canRefract = refract(-rOrigine.direction, n, 1.5, refracted);
    Ray rRefracted{rReflect.origin-rReflect.direction*0.1f+refracted*0.1f, refracted};
    if(canRefract) {
        float u = random_u();
        if(u < fresnel)
            return radiance(rReflect, countdown);
        else
            return radiance(rRefracted, countdown);
    }
        //return fresnel*radiance(rReflect, countdown)+(1-fresnel)*radiance(rRefracted, countdown);
    else
        return fresnel*radiance(rReflect, countdown);
}

glm::vec3 indirect(const Ray &rOrigine, const Ray &rReflect, const glm::vec3 & n, int countdown, const Mirror &mirror) {
    return radiance(rReflect, countdown);
}

int main (int, char **)
{
    int w = 768, h = 768;
    std::vector<glm::vec3> colors(w * h, glm::vec3{0.f, 0.f, 0.f});

    Ray cam {{50, 52, 295.6}, glm::normalize(glm::vec3{0, -0.042612, -1})};	// cam pos, dir
    float near = 1.f;
    float far = 10000.f;

    glm::mat4 camera =
        glm::scale(glm::mat4(1.f), glm::vec3(float(w), float(h), 1.f))
        * glm::translate(glm::mat4(1.f), glm::vec3(0.5, 0.5, 0.f))
        * glm::perspective(float(54.5f * pi / 180.f), float(w) / float(h), near, far)
        * glm::lookAt(cam.origin, cam.origin + cam.direction, glm::vec3(0, 1, 0))
        ;

    glm::mat4 screenToRay = glm::inverse(camera);

    #pragma omp parallel for
    for (int y = 0; y < h; y++)
    {
        std::cerr << "\rRendering: " << 100 * y / (h - 1) << "%";

        for (unsigned short x = 0; x < w; x++)
        {
            glm::vec3 r;
            float smoothies = 120.f;
            for(int smooths = 0; smooths < smoothies; ++smooths)
            {
                float u = random_u();
                float v = random_u();
                float R = sqrt(-2*log(u));
                float xDecal = R * cos(2*pi*u);
                float yDecal = R * sin(2*pi*v);
                glm::vec4 p0 = screenToRay * glm::vec4{float(x)+xDecal, float(h - y )+ yDecal, 0.f, 1.f};
                glm::vec4 p1 = screenToRay * glm::vec4{float(x)+xDecal, float(h - y )+ yDecal, 1.f, 1.f};

                glm::vec3 pp0 = glm::vec3(p0 / p0.w);
                glm::vec3 pp1 = glm::vec3(p1 / p1.w);

                glm::vec3 d = glm::normalize(pp1 - pp0);

                r += radiance (Ray{pp0, d});

            }
            r/=smoothies;
            colors[y * w + x] += glm::clamp(r, glm::vec3(0.f, 0.f, 0.f), glm::vec3(1.f, 1.f, 1.f)) * 0.25f;
        }
    }

    {
        std::fstream f("C:\\Users\\etu\\Desktop\\image.ppm", std::fstream::out);
        f << "P3\n" << w << " " << h << std::endl << "255" << std::endl;

        for (auto c : colors)
            f << toInt(c.x) << " " << toInt(c.y) << " " << toInt(c.z) << " ";
    }
}

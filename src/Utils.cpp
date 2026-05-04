#include "Math/Utils.hpp"
#include "G_Objects.hpp"

namespace Math
{
    float GRAVITY = -9.80f;   

    void InitializePyramidAxes(std::shared_ptr<G_Objects::P_Objects> pyramid, int baseSides) {
        std::vector<float>& vertices = pyramid->vertices;
        Math::Vector3 Ace = GetVertsFromVertices(vertices, baseSides);
        pyramid->localNormals.push_back(Math::Vector3(0.0f, -1.0f, 0.0f));

        for (int i = 0; i < baseSides; ++i) {
            Math::Vector3 vCurr = GetVertsFromVertices(vertices, i);
            Math::Vector3 vNext = GetVertsFromVertices(vertices, (i + 1) % baseSides);

            Math::Vector3 sideNormal = (vNext - vCurr).cross(Ace - vCurr).normalize();
            pyramid->localNormals.push_back(sideNormal);
            pyramid->localEdges.push_back(vNext - vCurr);
            pyramid->localEdges.push_back(Ace - vCurr);
        }
    }
}

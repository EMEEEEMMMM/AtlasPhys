#pragma once

#include "G_Objects.hpp"
#include "Math/Matrix4.hpp"
#include "Math/Vector3.hpp"
#include "Math/Utils.hpp"

#include <vector>

namespace SAT {
    struct Range { float min, max; };

    struct CollisionInfo {
        bool collided = false;
        float penetration = 1e10f;
        Math::Vector3 normal;
    };

    Range GetProjectionRange(std::shared_ptr<G_Objects::P_Objects> obj, const Math::Vector3& axis) {
        Math::Matrix4 model = obj->get_model_matrix();
        float val = (model * Math::GetVertsFromVertices(obj->vertices, 0)).dot(axis);
        Range r = {val ,val};

        int vertexCount = obj->vertices.size() / 7;
        for (int i = 1; i < vertexCount; ++i) {
            float point = (model * Math::GetVertsFromVertices(obj->vertices, i)).dot(axis);
            if (point < r.min) r.min = point;
            if (point > r.max) r.max = point;
        }

        return r;
    }

    bool Overlap(Range r1, Range r2) {
        return (r1.max >= r2.min && r2.max >= r1.min);
    }

    CollisionInfo collision_check(std::shared_ptr<G_Objects::P_Objects> objA, std::shared_ptr<G_Objects::P_Objects> objB) {
        CollisionInfo info;
        std::vector<Math::Vector3> axes;
        
        Math::Matrix4 modelA = objA->get_model_matrix();
        Math::Matrix4 modelB = objB->get_model_matrix();

        for (auto& n : objA->localNormals) axes.push_back(modelA.TransformVector(n).normalize());
        for (auto& n : objB->localNormals) axes.push_back(modelB.TransformVector(n).normalize());

        for (auto& edgeA : objA->localEdges) {
            Math::Vector3 worldEdgeA = modelA.TransformVector(edgeA);
            for (auto& edgeB : objB->localEdges) {
                Math::Vector3 worldEdgeB = modelB.TransformVector(edgeB);
                Math::Vector3 crossAxis = worldEdgeA.cross(worldEdgeB);

                if (crossAxis.length() > 1e-5f) {
                    axes.push_back(crossAxis.normalize());
                }
            }
        }

        for (const auto& axis : axes) {
            Range rangeA = GetProjectionRange(objA, axis);
            Range rangeB = GetProjectionRange(objB, axis);

            float overlap = std::min(rangeA.max, rangeB.max) - std::max(rangeA.min, rangeB.min);

            if (overlap <= 0) return CollisionInfo();

            if (overlap < info.penetration) {
                info.penetration = overlap;
                info.normal = axis;
            }
        }

        Math::Vector3 centerA = {modelA.m[3][0], modelA.m[3][1], modelA.m[3][2]};
        Math::Vector3 centerB = {modelB.m[3][0], modelB.m[3][1], modelB.m[3][2]};
        if (info.normal.dot(centerA - centerB) < 0) {
            info.normal = info.normal * -1.0f;
        }

        info.collided = true;
        return info;
    }

    void ResolveCollisionXPBD(
        std::shared_ptr<G_Objects::P_Objects> objA,
        std::shared_ptr<G_Objects::P_Objects> objB,
        const CollisionInfo& info) {
            if (!info.collided) return;

            float wA = (objA->mass > 0) ? (1.0f / objA->mass) : 0.0f;
            float wB = (objB->mass > 0) ? (1.0f / objB->mass) : 0.0f;

            float wSum = wA + wB;
            if (wSum <= 0) return;

            Math::Vector3 correction = info.normal * (info.penetration / wSum);

            if (wA > 0) {
                objA->position = objA->position + (correction * wA);                
            }

            if (wB > 0) {
                objB->position = objB->position - (correction * wB);                
            }

            Math::Vector3 relVelocity = objA->velocity - objB->velocity;
            float velocityNormal = relVelocity.dot(info.normal);

            if (velocityNormal < 0) {
                float e = std::min(objA->restitution, objB->restitution);
                Math::Vector3 impulse = info.normal * (-(1.0f + e) * velocityNormal / wSum);

                objA->velocity = objA->velocity + (impulse * wA);
                objB->velocity = objB->velocity - (impulse * wB);
            }

        }
}

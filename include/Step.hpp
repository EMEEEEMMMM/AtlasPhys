#pragma once

#include "G_Objects.hpp"
#include "SAT.hpp"

#include <vector>
#include <memory>

const int SUBSTEPS = 5;

namespace Step{
    void integrate_prediction(std::shared_ptr<G_Objects::P_Objects> obj, float deltaTime) {
        obj->oldPosition = obj->position;
        obj->velocity += obj->acceleration * deltaTime;
        obj->position += obj->velocity * deltaTime;
    }

    void collision(std::vector<std::weak_ptr<G_Objects::P_Objects>>& objs) {
        if (objs.size() < 2) return;
        const std::size_t n = objs.size();
        for (int m = 0; m < 5; ++m) {
            for (std::size_t i = 0; i < n; ++i) {
                auto aPtr = objs[i].lock();
                if (!aPtr) continue;
                for (std::size_t j = i + 1; j < n; ++j) {
                    auto bPtr = objs[j].lock();
                    if (!bPtr) continue;
                    SAT::CollisionInfo info = SAT::collision_check(aPtr, bPtr);
                    // printf("info.collided: %d\n", info.collided);
                    // printf("info.normal: (%f, %f, %f)\n", info.normal.x, info.normal.y, info.normal.z);
                    // printf("info.penetration: %f\n", info.penetration);
                    if (info.collided) SAT::ResolveCollisionXPBD(aPtr, bPtr, info);
                }
            }
        }
    }

    void pStep(std::vector<std::weak_ptr<G_Objects::P_Objects>>& objs, float deltaTime) {
        float subDeltaTime = deltaTime / SUBSTEPS;

        for (int s = 0; s < SUBSTEPS; ++s) {
            for (auto& weak : objs) {
                if (auto obj = weak.lock()) if (obj->mass > 0) integrate_prediction(obj, subDeltaTime);
            }

            collision(objs);

            for (auto& weak : objs) {
                if (auto obj = weak.lock()) if (obj->mass > 0) obj->velocity = (obj->position - obj->oldPosition) / subDeltaTime;
            }
        }
    }
}
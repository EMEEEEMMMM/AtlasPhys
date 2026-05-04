#pragma once

#include "G_Objects.hpp"
#include "SAT.hpp"

#include <vector>
#include <memory>

namespace Step{
    void integrator(std::vector<std::weak_ptr<G_Objects::P_Objects>>& objs, float deltaTime) {
        for (auto it = objs.begin(); it != objs.end(); ) {
            if (auto obj = it->lock()) {
                obj->velocity += obj->acceleration * deltaTime;
                obj->position += obj->velocity * deltaTime;
                ++it;
            } else {
                it = objs.erase(it);
            }
        }
    }

    void collision(std::vector<std::weak_ptr<G_Objects::P_Objects>>& objs) {
        if (objs.size() < 2) return;
        printf("collision() called\n");
        const std::size_t n = objs.size();
        for (std::size_t i = 0; i < n; ++i) {
            auto aPtr = objs[i].lock();
            if (!aPtr) continue;
            for (std::size_t j = i + 1; j < n; ++j) {
                auto bPtr = objs[j].lock();
                if (!bPtr) continue;
                for (int k = 0; k < 4; ++k) {
                    SAT::CollisionInfo info = SAT::collision_check(aPtr, bPtr);
                    printf("info.collided: %d\n", info.collided);
                    SAT::ResolveCollisionXPBD(aPtr, bPtr, info);
                }
            }
        }
    }
}
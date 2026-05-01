#pragma once

#include "G_Objects.hpp"

#include <vector>
#include <memory>

namespace Step{
    void integrator(std::vector<std::weak_ptr<G_Objects::P_Objects>>& objs, float deltaTime) {
        for (auto it = objs.begin(); it != objs.end(); ) {
            if (auto obj = it->lock()) {
                obj->velocity += obj->acceleration * deltaTime;
                obj->position += obj->velocity * deltaTime;
                ++it;
            } 
        }
    }
}
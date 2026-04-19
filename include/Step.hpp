#pragma once

#include "G_Objects.hpp"

#include <vector>

namespace Step{
    void integrator(std::vector<G_Objects::P_Objects>& g_Objects, float deltaTime) {
        for (G_Objects::P_Objects& obj: g_Objects) {
            obj.velocity += obj.acceleration * deltaTime;
            obj.position += obj.velocity * deltaTime;
        }
    }
}
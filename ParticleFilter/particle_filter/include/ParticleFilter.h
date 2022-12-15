#pragma once

#include <memory>
#include <random>
#include <string>
#include <unordered_map>

#include <landmarkSim2dLib/Sim.h>

class ParticleFilter
{
  public:
    struct Particle
    {
        explicit Particle(const int16_t &id, const landmarkSim2D::Pose2D &pose, const float &weight)
            : id{id}, pose{pose}, weight{weight}
        {
        }

        int16_t               id;
        landmarkSim2D::Pose2D pose;
        float                 weight;
    };

    ParticleFilter(const std::string &mapFilePath, const int16_t &numParticles, const float &sensorRange);
    void PredictAndExplore(const landmarkSim2D::ControlInput &ctrlInput, const double &dt);
    void UpdateWeightsWithObservations(const std::vector<landmarkSim2D::RangeBearingObs> &landmarkObservations);
    std::unordered_map<size_t, size_t>
    AssociateObservationsToParticleLandmarks(const std::vector<landmarkSim2D::Map::Landmark> &robotObservations,
                                             const std::vector<landmarkSim2D::Map::Landmark> &particleLandmarks);
    landmarkSim2D::Map::Landmark
    TransformRangeBearingObsToMapFrame(const landmarkSim2D::RangeBearingObs &rangeBearingObs,
                                       const landmarkSim2D::Pose2D          &poseInMapFrame);
    std::vector<landmarkSim2D::Map::Landmark>
    TransformObservationsToLandmarksInMapFrame(const std::vector<landmarkSim2D::RangeBearingObs> &landmarkObservations,
                                               const landmarkSim2D::Pose2D                       &poseInMapFrame);
    void UpdateParticleWeight(Particle                                        &particle,
                              const std::vector<landmarkSim2D::Map::Landmark> &robotLandmarkObservationsInMapFrame,
                              const std::vector<landmarkSim2D::Map::Landmark> &particleLandmarks,
                              const std::unordered_map<size_t, size_t>        &obsLmToParLmMap);

    std::vector<Particle> particles_;

  private:
    std::unique_ptr<landmarkSim2D::Map> map_;

    // TODO: Using same engine for all particles and variables causes correlation of noise?
    std::random_device randDev_{};
    std::mt19937       randGenEngine_{randDev_()};

    float sigmaPosX_; // standard dev for the exploration noise [m]
    float sigmaPosY_; // standard dev for the exploration noise [m]
    float sigmaYaw_;  // standard dev for the exploration noise [m]

    float sensorRange_;
};
#pragma once

#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>

#include <LandmarkSim2dLib/Sim.h>

class ParticleFilter
{
  public:
    struct Particle
    {
        Particle() = default;
        explicit Particle(const int16_t &id, const landmarkSim2D::Pose2D &pose, const double &weight)
            : id{id}, pose{pose}, weight{weight}
        {
        }

        int16_t               id;
        landmarkSim2D::Pose2D pose;
        double                weight;
        double beliefError2{0}; // avg. Eucledian SQUARED distance between particle landmark belief vs robot observation
    };

    ParticleFilter(const std::string &mapFilePath, const int16_t &numParticles, const double &sensorRange);
    void PredictAndExplore(const landmarkSim2D::ControlInput &ctrlInput, const double &dt);
    void UpdateWeightsWithObservations(const std::vector<landmarkSim2D::RangeBearingObs> &landmarkObservations);
    void UpdateWeightsWithObservations2(const std::vector<landmarkSim2D::RangeBearingObs> &landmarkObservations);
    std::unordered_map<size_t, size_t>
    AssociateObservationsToParticleLandmarks(const std::vector<landmarkSim2D::Map::Landmark> &robotObservations,
                                             const std::vector<landmarkSim2D::Map::Landmark> &particleLandmarks);
    std::unordered_map<size_t, size_t> AssociateObservationsToParticleObservations(
        const std::vector<landmarkSim2D::RangeBearingObs> &robotObservations,
        const std::vector<landmarkSim2D::RangeBearingObs> &particleObservations);
    landmarkSim2D::Map::Landmark
    TransformRangeBearingObsToMapFrame(const landmarkSim2D::RangeBearingObs &rangeBearingObs,
                                       const landmarkSim2D::Pose2D          &poseInMapFrame);
    std::vector<landmarkSim2D::Map::Landmark>
    TransformObservationsToLandmarksInMapFrame(const std::vector<landmarkSim2D::RangeBearingObs> &landmarkObservations,
                                               const landmarkSim2D::Pose2D                       &poseInMapFrame);
    void UpdateParticleWeightEuclideanDist(
        Particle                                        &particle,
        const std::vector<landmarkSim2D::Map::Landmark> &robotLandmarkObservationsInMapFrame,
        const std::vector<landmarkSim2D::Map::Landmark> &particleLandmarks,
        const std::unordered_map<size_t, size_t>        &obsLmToParLmMap);
    void UpdateParticleWeight2(Particle                                          &particle,
                               const std::vector<landmarkSim2D::RangeBearingObs> &robotObservations,
                               const std::vector<landmarkSim2D::RangeBearingObs> &particleObservations,
                               const std::unordered_map<size_t, size_t>          &mapRobObsToParObs);
    void UpdateParticleWeightMultivariateGaussian(
        Particle                                        &particle,
        const std::vector<landmarkSim2D::Map::Landmark> &robotLandmarkObservationsInMapFrame,
        const std::vector<landmarkSim2D::Map::Landmark> &particleLandmarks,
        const std::unordered_map<size_t, size_t>        &obsLmToParLmMap);

    std::vector<landmarkSim2D::RangeBearingObs> GenerateObservations(const Particle &particle);
    void                                        ResampleParticles();
    void                                        CheckFilterReset();
    void                                        ResetFilter();

    std::vector<Particle>                               particles_;
    std::multimap<double, size_t, std::greater<double>> bestParticles_; // Top 10 % of the particles by weight

  private:
    std::unique_ptr<landmarkSim2D::Map> map_;
    landmarkSim2D::Map::BboxExtent      mapExtent_;

    // TODO: Using same engine for all particles and variables causes correlation of noise?
    std::random_device randDev_{};
    std::mt19937       randGenEngine_{randDev_()};

    double sigmaPosX_; // standard dev for the exploration noise [m]
    double sigmaPosY_; // standard dev for the exploration noise [m]
    double sigmaYaw_;  // standard dev for the exploration noise [m]

    double sigmaFilterX_; // sigma used in weight update
    double sigmaFilterY_;

    double sensorRange_;
    double avgBeliefError2_; // average squared belief error of all particles
    double filterResetThresh2_;
};
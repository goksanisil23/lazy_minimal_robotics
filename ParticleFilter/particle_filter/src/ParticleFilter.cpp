#include "ParticleFilter.h"

#include <cstdlib>
#include <ctime>

ParticleFilter::ParticleFilter(const std::string &mapFilePath, const int16_t &numParticles, const float &sensorRange)
    // : sigmaPosX_{0.3}, sigmaPosY_{0.3}, sigmaYaw_{0.01}
    : sigmaPosX_{0.1}, sigmaPosY_{0.1}, sigmaYaw_{0.01}, sensorRange_{sensorRange}
{
    map_ = std::make_unique<landmarkSim2D::Map>(mapFilePath);

    // Based on the distribution of the landmarks in the map, we evenly spread out particles during initialization
    // 1) Get the bounding box surrounding the landmarks
    // 2) Create N particles within the bounding box, randomly spread
    landmarkSim2D::Map::BboxExtent mapExtent = map_->GetBoundingExtentOfMap();

    std::srand(std::time(nullptr)); // use current time as seed for random generator

    for (int16_t idxParticle = 0; idxParticle < numParticles; idxParticle++)
    {
        float randX = mapExtent.lowX +
                      static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (mapExtent.highX - mapExtent.lowX)));

        float randY = mapExtent.lowY +
                      static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (mapExtent.highY - mapExtent.lowY)));

        float randYaw = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (2 * M_PI));

        landmarkSim2D::Pose2D initPose(randX, randY, randYaw);
        particles_.push_back(Particle(idxParticle, initPose, 1.0));
    }
}

void ParticleFilter::PredictAndExplore(const landmarkSim2D::ControlInput &ctrlInput, const double &dt)
{
    for (auto &particle : particles_)
    {
        // 1) dead-reckon with the motion model + ctrl input
        landmarkSim2D::Pose2D predictedPose = landmarkSim2D::Robot::IterateMotionModel(particle.pose, dt, ctrlInput);

        // 2) Add exploration noise
        std::normal_distribution<float> normDistPosX{0, sigmaPosX_};
        std::normal_distribution<float> normDistPosY{0, sigmaPosY_};
        std::normal_distribution<float> normDistYaw{0, sigmaYaw_};
        predictedPose.posX += normDistPosX(randGenEngine_);
        predictedPose.posY += normDistPosY(randGenEngine_);
        predictedPose.yawRad += normDistYaw(randGenEngine_);

        particle.pose = predictedPose;
    }
}

void ParticleFilter::UpdateWeightsWithObservations(
    const std::vector<landmarkSim2D::RangeBearingObs> &landmarkObservations)
{
    for (auto &particle : particles_)
    {
        // 1) Find the landmarks within the range of this particle
        std::vector<landmarkSim2D::Map::Landmark> particleLandmarks{
            map_->GetLandmarksWithinRadius(particle.pose, sensorRange_)};

        // 2) Transform each robot observation to map frame, using this particle's predicted state
        std::vector<landmarkSim2D::Map::Landmark> robotLandmarkObservationsInMapFrame{
            TransformObservationsToLandmarksInMapFrame(landmarkObservations, particle.pose)};

        if (!particleLandmarks.empty())
        {
            // 3) Associate robot's observed landmarks to particle's predicted landmarks
            // This step assigns landmark id to robot's landmark observation
            std::unordered_map<size_t, size_t> obsLmToParLm{
                AssociateObservationsToParticleLandmarks(robotLandmarkObservationsInMapFrame, particleLandmarks)};

            // 4) Update the weight
            // Each observation contributes to weight
            // Use distance between the associated particle landmark and robot landmark
            // particle.weight = 1.0f; // reset
            particle.weight = 0.0f; // reset
            UpdateParticleWeight(particle, robotLandmarkObservationsInMapFrame, particleLandmarks, obsLmToParLm);
        }
        else
        {
            particle.weight = std::numeric_limits<float>::min();
        }
    }
}

std::unordered_map<size_t, size_t> ParticleFilter::AssociateObservationsToParticleLandmarks(
    const std::vector<landmarkSim2D::Map::Landmark> &robotObservations,
    const std::vector<landmarkSim2D::Map::Landmark> &particleLandmarks)
{
    std::unordered_map<size_t, size_t> obsLmToParLm; // observation landmark to particle landmark mapping

    for (size_t robObsIdx = 0; robObsIdx < robotObservations.size(); robObsIdx++)
    {
        const auto &robotObsLm = robotObservations.at(robObsIdx);
        float       minDistance2{std::numeric_limits<float>::max()}; // distance squared
        size_t      bestPartLmIdx;
        for (size_t partLmIdx = 0; partLmIdx < particleLandmarks.size(); partLmIdx++)
        {
            const auto &particleLm              = particleLandmarks.at(partLmIdx);
            auto        particleLmToRobObsDist2 = particleLm.Distance2(robotObsLm);
            if (particleLmToRobObsDist2 < minDistance2)
            {
                minDistance2  = particleLmToRobObsDist2;
                bestPartLmIdx = partLmIdx;
            }
        }
        obsLmToParLm.insert(std::pair<size_t, size_t>(robObsIdx, bestPartLmIdx));
    }

    return obsLmToParLm;
}

landmarkSim2D::Map::Landmark
ParticleFilter::TransformRangeBearingObsToMapFrame(const landmarkSim2D::RangeBearingObs &rangeBearingObs,
                                                   const landmarkSim2D::Pose2D          &poseInMapFrame)
{
    // 1) Create landmark in robot frame
    landmarkSim2D::Map::Landmark landmarkInRobotFrame{-1,
                                                      rangeBearingObs.range * cos(rangeBearingObs.angleRad),
                                                      rangeBearingObs.range * sin(rangeBearingObs.angleRad)};
    // 2) Transform landmark from robot frame to map frame
    // SE(2) homogenous transformation matrix
    // Θ = from map axis to robot axis
    // | cos(Θ) -sin(Θ) t_from_map_to_robot_x |
    // | sin(Θ)  cos(Θ) t_from_map_to_robot_y |
    // |   0       0              1           |
    landmarkSim2D::Map::Landmark landmarkInMapFrame;
    landmarkInMapFrame.posX = cos(poseInMapFrame.yawRad) * landmarkInRobotFrame.posX -
                              sin(poseInMapFrame.yawRad) * landmarkInRobotFrame.posY + poseInMapFrame.posX;
    landmarkInMapFrame.posY = sin(poseInMapFrame.yawRad) * landmarkInRobotFrame.posX +
                              cos(poseInMapFrame.yawRad) * landmarkInRobotFrame.posY + poseInMapFrame.posY;

    return landmarkInMapFrame;
}

std::vector<landmarkSim2D::Map::Landmark> ParticleFilter::TransformObservationsToLandmarksInMapFrame(
    const std::vector<landmarkSim2D::RangeBearingObs> &landmarkObservations,
    const landmarkSim2D::Pose2D                       &poseInMapFrame)
{
    std::vector<landmarkSim2D::Map::Landmark> robotLandmarkObservationsInMapFrame;
    for (const auto &lmObs : landmarkObservations)
    {
        robotLandmarkObservationsInMapFrame.push_back(
            landmarkSim2D::Map::Landmark{TransformRangeBearingObsToMapFrame(lmObs, poseInMapFrame)});
    }

    return robotLandmarkObservationsInMapFrame;
}

void ParticleFilter::UpdateParticleWeight(
    Particle                                        &particle,
    const std::vector<landmarkSim2D::Map::Landmark> &robotLandmarkObservationsInMapFrame,
    const std::vector<landmarkSim2D::Map::Landmark> &particleLandmarks,
    const std::unordered_map<size_t, size_t>        &obsLmToParLmMap)
{
    for (size_t robObsIdx = 0; robObsIdx < robotLandmarkObservationsInMapFrame.size(); robObsIdx++)
    {
        const auto &associatedParticleLmIdx{obsLmToParLmMap.at(robObsIdx)};
        const auto &associatedParticleLm{particleLandmarks.at(associatedParticleLmIdx)};
        auto        distance2 = robotLandmarkObservationsInMapFrame.at(robObsIdx).Distance2(associatedParticleLm);
        particle.weight += 1 / distance2;
    }
}
#include "ParticleFilter.h"

#include "TimeUtil.h"

#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iomanip>

namespace
{
// after how many iterations of large particle error should the filter reset
const size_t kFilterResetCtrLimit{20};
} // namespace

ParticleFilter::ParticleFilter(const std::string &mapFilePath, const int16_t &numParticles, const double &sensorRange)
    : sigmaPosX_{0.3}, sigmaPosY_{0.3}, sigmaYaw_{0.01}, sensorRange_{sensorRange}
// : sigmaPosX_{0.04}, sigmaPosY_{0.04}, sigmaYaw_{0.01}, sensorRange_{sensorRange}
// : sigmaPosX_{0.15}, sigmaPosY_{0.15}, sigmaYaw_{0.04}, sensorRange_{sensorRange}, avgBeliefError2_{0}
{
    map_                = std::make_unique<landmarkSim2D::Map>(mapFilePath);
    mapExtent_          = map_->GetBoundingExtentOfMap();
    particles_          = std::vector<Particle>(numParticles);
    filterResetThresh2_ = std::pow(sensorRange_ * 0.5, 2);
    sigmaFilterX_       = sigmaPosX_;
    sigmaFilterY_       = sigmaPosY_;
    ResetFilter();
}

void ParticleFilter::ResetFilter()
{
    // Based on the distribution of the landmarks in the map, we evenly spread out particles during initialization
    // 1) Get the bounding box surrounding the landmarks
    // 2) Create N particles within the bounding box, randomly spread
    std::srand(std::time(nullptr)); // use current time as seed for random generator

    for (size_t idxParticle = 0; idxParticle < particles_.size(); idxParticle++)
    {
        float randX = mapExtent_.lowX + static_cast<float>(rand()) /
                                            (static_cast<float>(RAND_MAX / (mapExtent_.highX - mapExtent_.lowX)));

        float randY = mapExtent_.lowY + static_cast<float>(rand()) /
                                            (static_cast<float>(RAND_MAX / (mapExtent_.highY - mapExtent_.lowY)));

        float randYaw = static_cast<float>(rand()) / static_cast<float>(RAND_MAX / (2 * M_PI));

        landmarkSim2D::Pose2D initPose(randX, randY, randYaw); // random init pose
        // landmarkSim2D::Pose2D initPose(12.5f, 0.0f, M_PI / 2.0f); // correct init pose
        // landmarkSim2D::Pose2D initPose(randX, randY, M_PI / 2.0f); // random position, correct yaw
        particles_.at(idxParticle) = Particle(idxParticle, initPose, 1.0);
    }
    avgBeliefError2_ = 0.0;
}

void ParticleFilter::PredictAndExplore(const landmarkSim2D::ControlInput &ctrlInput, const double &dt)
{
    for (auto &particle : particles_)
    {
        // 1) dead-reckon with the motion model + ctrl input
        landmarkSim2D::Pose2D predictedPose = landmarkSim2D::Robot::IterateMotionModel(particle.pose, dt, ctrlInput);

        // 2) Add exploration noise (also called stochastic diffusion)
        std::normal_distribution<double> normDistPosX{0, sigmaPosX_};
        std::normal_distribution<double> normDistPosY{0, sigmaPosY_};
        std::normal_distribution<double> normDistYaw{0, sigmaYaw_};
        predictedPose.posX += normDistPosX(randGenEngine_);
        predictedPose.posY += normDistPosY(randGenEngine_);
        predictedPose.yawRad += normDistYaw(randGenEngine_);

        particle.pose = predictedPose;
    }
}

void ParticleFilter::UpdateWeightsWithObservations(
    const std::vector<landmarkSim2D::RangeBearingObs> &landmarkObservations)
{
    // Reset the avg belief error of filter
    avgBeliefError2_ = 0.0f;

    for (auto &particle : particles_)
    {
        // 1) Find the landmarks within the range of this particle
        std::vector<landmarkSim2D::Map::Landmark> particleLandmarks{
            map_->GetLandmarksWithinRadius(particle.pose, sensorRange_)};

        if (!particleLandmarks.empty())
        {
            // 2) Transform each robot observation to map frame, using this particle's predicted state
            std::vector<landmarkSim2D::Map::Landmark> robotLandmarksInMapFrame{
                TransformObservationsToLandmarksInMapFrame(landmarkObservations, particle.pose)};

            // 3) Associate robot's observed landmarks to particle's predicted landmarks
            // This step assigns landmark id to robot's landmark observation
            std::unordered_map<size_t, size_t> obsLmToParLm{
                AssociateObservationsToParticleLandmarks(robotLandmarksInMapFrame, particleLandmarks)};

            // 4) Update the weight
            // Each observation contributes to weight
            // Use distance between the associated particle landmark and robot landmark
            // particle.weight = 0.0f; // reset
            // UpdateParticleWeightEuclideanDist(particle, robotLandmarksInMapFrame, particleLandmarks, obsLmToParLm);
            particle.weight = 1.0; // reset
            UpdateParticleWeightMultivariateGaussian(
                particle, robotLandmarksInMapFrame, particleLandmarks, obsLmToParLm);
        }
        else
        {
            // No landmarks in the vicinity of the particle, worst case assigned
            particle.weight       = std::numeric_limits<float>::min();
            particle.beliefError2 = filterResetThresh2_;
            avgBeliefError2_ += particle.beliefError2;
        }
    }
    ResampleParticles();
    avgBeliefError2_ /= static_cast<double>(particles_.size());
}

void ParticleFilter::UpdateWeightsWithObservations2(
    const std::vector<landmarkSim2D::RangeBearingObs> &landmarkObservations)
{
    for (auto &particle : particles_)
    {
        // 1) Generate observations from particle's p.o.v
        std::vector<landmarkSim2D::RangeBearingObs> particleObservations{GenerateObservations(particle)};

        if (!particleObservations.empty())
        {
            // 2) Associate robot's observations to particle's predicted observations
            std::unordered_map<size_t, size_t> mapRobObsToParObs{
                AssociateObservationsToParticleObservations(landmarkObservations, particleObservations)};

            // 3) Update the weight
            // Each observation contributes to weight
            // Use normalized error between particle observation and robot observation
            // particle.weight = 1.0f; // reset
            particle.weight = 0.0f; // reset
            UpdateParticleWeight2(particle, landmarkObservations, particleObservations, mapRobObsToParObs);
        }
        else
        {
            particle.weight = std::numeric_limits<double>::min();
        }
    }
    ResampleParticles();
}

std::unordered_map<size_t, size_t> ParticleFilter::AssociateObservationsToParticleObservations(
    const std::vector<landmarkSim2D::RangeBearingObs> &robotObservations,
    const std::vector<landmarkSim2D::RangeBearingObs> &particleObservations)
{
    std::unordered_map<size_t, size_t> mapRobotObsToParObs; // robot observation to particle observation mapping

    for (size_t robObsIdx = 0; robObsIdx < robotObservations.size(); robObsIdx++)
    {
        const auto &robotObs = robotObservations.at(robObsIdx);
        double      minError{std::numeric_limits<double>::max()};
        size_t      bestPartObsIdx{0};
        for (size_t partObsIdx = 0; partObsIdx < particleObservations.size(); partObsIdx++)
        {
            const auto &particleObs                     = particleObservations.at(partObsIdx);
            auto        particleObsRangeToRobotObsRange = std::abs(robotObs.range - particleObs.range);
            auto        particleObsBearToRobotObsBear   = std::abs(robotObs.angleRad - particleObs.angleRad);
            // Normalize the errors
            double normRangeError = particleObsRangeToRobotObsRange / sensorRange_;
            double normBearError  = particleObsBearToRobotObsBear / (2.0 * M_PI);
            double normError      = normBearError * normRangeError;
            if (normError < minError)
            {
                minError       = normError;
                bestPartObsIdx = partObsIdx;
            }
        }
        mapRobotObsToParObs.insert(std::pair<size_t, size_t>(robObsIdx, bestPartObsIdx));
    }

    return mapRobotObsToParObs;
}

std::unordered_map<size_t, size_t> ParticleFilter::AssociateObservationsToParticleLandmarks(
    const std::vector<landmarkSim2D::Map::Landmark> &robotObservations,
    const std::vector<landmarkSim2D::Map::Landmark> &particleLandmarks)
{
    std::unordered_map<size_t, size_t> mapObsLmToParLm; // observation landmark to particle landmark mapping

    for (size_t robObsIdx = 0; robObsIdx < robotObservations.size(); robObsIdx++)
    {
        const auto &robotObsLm = robotObservations.at(robObsIdx);
        double      minDistance2{std::numeric_limits<double>::max()}; // distance squared
        size_t      bestPartLmIdx{0};
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
        mapObsLmToParLm.insert(std::pair<size_t, size_t>(robObsIdx, bestPartLmIdx));
    }

    return mapObsLmToParLm;
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

void ParticleFilter::UpdateParticleWeightEuclideanDist(
    Particle                                        &particle,
    const std::vector<landmarkSim2D::Map::Landmark> &robotLandmarkObservationsInMapFrame,
    const std::vector<landmarkSim2D::Map::Landmark> &particleLandmarks,
    const std::unordered_map<size_t, size_t>        &obsLmToParLmMap)
{
    particle.beliefError2 = 0.0; // reset belief error
    for (size_t robObsIdx = 0; robObsIdx < robotLandmarkObservationsInMapFrame.size(); robObsIdx++)
    {
        const auto &associatedParticleLmIdx{obsLmToParLmMap.at(robObsIdx)};
        const auto &associatedParticleLm{particleLandmarks.at(associatedParticleLmIdx)};
        auto        distance2 = robotLandmarkObservationsInMapFrame.at(robObsIdx).Distance2(associatedParticleLm);
        particle.weight += 1.0f / distance2;
        particle.beliefError2 += distance2;
    }
    particle.beliefError2 /= static_cast<double>(robotLandmarkObservationsInMapFrame.size());
    avgBeliefError2_ += particle.beliefError2;
}

void ParticleFilter::UpdateParticleWeight2(Particle                                          &particle,
                                           const std::vector<landmarkSim2D::RangeBearingObs> &robotObservations,
                                           const std::vector<landmarkSim2D::RangeBearingObs> &particleObservations,
                                           const std::unordered_map<size_t, size_t>          &mapRobObsToParObs)
{
    for (size_t robObsIdx = 0; robObsIdx < robotObservations.size(); robObsIdx++)
    {
        const auto &associatedParticleObsIdx{mapRobObsToParObs.at(robObsIdx)};
        const auto &associatedParticleObs{particleObservations.at(associatedParticleObsIdx)};

        // Distance score
        auto particleObsRangeToRobotObsRange =
            std::abs(robotObservations.at(robObsIdx).range - associatedParticleObs.range);
        auto particleObsBearToRobotObsBear =
            std::abs(robotObservations.at(robObsIdx).angleRad - associatedParticleObs.angleRad);
        // Normalize the errors
        double normRangeError = particleObsRangeToRobotObsRange / sensorRange_;
        double normBearError  = particleObsBearToRobotObsBear / (2.0 * M_PI);
        double normError      = normBearError * normRangeError;

        particle.weight += 1.0f / normError;
    }
}

// Idea is to represent this particle's probability (hence weight) by a multivariate Gaussian dist.
// particle's current landmark observation belief (X=sample)
// is normally distributed around the robot's actual observation (mu)
// I have used the exploration noise sigma for the distribution here. This is not very correct since ideally,
// landmark/sensor measurement noise sigma should've been used. Also assuming x & y are uncorrelated ...
void ParticleFilter::UpdateParticleWeightMultivariateGaussian(
    Particle                                        &particle,
    const std::vector<landmarkSim2D::Map::Landmark> &robotLandmarkObservationsInMapFrame,
    const std::vector<landmarkSim2D::Map::Landmark> &particleLandmarks,
    const std::unordered_map<size_t, size_t>        &obsLmToParLmMap)
{
    particle.beliefError2            = 0; // reset belief error
    static const double constant     = 1.0 / (2.0 * M_PI * sigmaFilterX_ * sigmaFilterY_);
    static const double expo_x_const = (2.0 * sigmaFilterX_ * sigmaFilterX_);
    static const double expo_y_const = (2.0 * sigmaFilterY_ * sigmaFilterY_);

    for (size_t robObsIdx = 0; robObsIdx < robotLandmarkObservationsInMapFrame.size(); robObsIdx++)
    {
        const auto &associatedParticleLmIdx{obsLmToParLmMap.at(robObsIdx)};
        const auto &associatedParticleLm{particleLandmarks.at(associatedParticleLmIdx)};
        auto        distanceX = associatedParticleLm.posX - robotLandmarkObservationsInMapFrame.at(robObsIdx).posX;
        auto        distanceY = associatedParticleLm.posY - robotLandmarkObservationsInMapFrame.at(robObsIdx).posY;
        double      expo      = std::exp(-(

            (distanceX * distanceX / (expo_x_const)) + (distanceY * distanceY / (expo_y_const))

                ));

        double weight = constant * expo;
        particle.weight *= weight;
        particle.beliefError2 += (distanceX * distanceX + distanceY * distanceY);
        // std::cout << std::fixed << std::setprecision(20) << "dx: " << distanceX << " dy: " << distanceY
        //           << " w: " << weight << std::endl;
    }
    // std::cout << std::fixed << std::setprecision(40) << "p: " << particle.id << " w: " << particle.weight << std::endl;

    particle.beliefError2 /= static_cast<double>(robotLandmarkObservationsInMapFrame.size());
    avgBeliefError2_ += particle.beliefError2;
}

void ParticleFilter::ResampleParticles()
{
    std::vector<Particle> resampledParticles;

    std::vector<double> cumSum(particles_.size() + 1, 0.0);
    for (size_t i = 0; i < particles_.size(); i++)
    {
        cumSum.at(i + 1) = cumSum.at(i) + particles_.at(i).weight;
    }

    double                                 cumSumLimit = cumSum.back();
    std::uniform_real_distribution<double> uniformDist(0.0, cumSumLimit);
    std::cout << "cumSumLimit " << cumSumLimit << std::endl;
    // throw std::runtime_error("error");

    for (size_t i = 0; i < particles_.size(); i++)
    {
        // Pick based on where the random number lies on the resample wheel
        double seed = uniformDist(randGenEngine_);
        // Find where to put the seed
        int j = (cumSum.size() - 1) - 1; // 1 element before last in cumsum vector
        while (j >= 0)
        {
            if (seed > cumSum.at(j))
            {
                resampledParticles.push_back(particles_.at(j));

                // update the best Particles list, with limited size
                bestParticles_.insert(std::pair<double, size_t>(particles_.at(j).weight, static_cast<size_t>(j)));
                if (bestParticles_.size() > (particles_.size() / 10))
                {
                    bestParticles_.erase(std::prev(bestParticles_.end()));
                }
                break;
            }
            j--;
        }
    }
    std::cout << std::fixed << std::setprecision(10) << "w1: " << particles_.at(0).weight
              << " w2: " << particles_.at(1).weight << " w3: " << particles_.at(2).weight << std::endl;
    std::cout << "resample size: " << resampledParticles.size() << std::endl;
    particles_ = resampledParticles;
}

std::vector<landmarkSim2D::RangeBearingObs> ParticleFilter::GenerateObservations(const Particle &particle)
{
    auto landmarksInRange(this->map_->GetLandmarksWithinRadius(particle.pose, this->sensorRange_));
    std::vector<landmarkSim2D::RangeBearingObs> particleObservations;
    int16_t                                     obsId = 0;
    for (const auto &landmarkInRange : landmarksInRange)
    {
        landmarkSim2D::RangeBearingObs observation;
        float                          angleToLandmark =
            atan2(landmarkInRange.posY - particle.pose.posY, landmarkInRange.posX - particle.pose.posX) -
            particle.pose.yawRad;
        observation.angleRad = angleToLandmark;
        observation.range =
            sqrt((landmarkInRange.posY - particle.pose.posY) * (landmarkInRange.posY - particle.pose.posY) +
                 (landmarkInRange.posX - particle.pose.posX) * (landmarkInRange.posX - particle.pose.posX));
        observation.id = obsId;

        particleObservations.push_back(observation);
        obsId++;
    }
    return particleObservations;
}

// If the average belief error of all particles are greater than sensor range for
// more than X iterations, we reset the filter within the map again
void ParticleFilter::CheckFilterReset()
{
    std::cout << "avg Belief error: " << avgBeliefError2_ << std::endl;
    static size_t resetCtr = 0;
    if (avgBeliefError2_ > filterResetThresh2_)
    {
        resetCtr++;
    }
    else
    {
        resetCtr = 0;
    }
    if (resetCtr > kFilterResetCtrLimit)
    {
        ResetFilter();
        resetCtr = 0;
        std::cout << "RESETING THE FILTER" << std::endl;
    }
}
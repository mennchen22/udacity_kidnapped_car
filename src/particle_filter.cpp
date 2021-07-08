/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>


using std::string;
using std::vector;
using std::normal_distribution;

void ParticleFilter::init(double x, double y, double theta, const double std[]) {
    /**
     * TODO: Set the number of particles. Initialize all particles to
     *   first position (based on estimates of x, y, theta and their uncertainties
     *   from GPS) and all weights to 1.
     * TODO: Add random Gaussian noise to each particle.
     * NOTE: Consult particle_filter.h for more information about this method
     *   (and others in this file).
     */
    num_particles = 100;  // TODO: Set the number of particles
    particles = vector<Particle>(num_particles);
    std::default_random_engine gen;

    double std_x, std_y, std_theta;  // Standard deviations for x, y, and theta

    // Set standard deviations for x, y, and theta
    std_x = std[0];
    std_y = std[1];
    std_theta = std[2];

    // This line creates a normal (Gaussian) distribution for x
    normal_distribution<double> dist_x(x, std_x);

    // Create normal distributions for y and theta
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; ++i) {
        // Sample from these normal distributions like this:
        Particle particle = Particle();
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particle.weight = 1;
        particles[i] = particle;
    }

    std::cout << "Init particle filter with " << num_particles << " particles with std x: " << std_x << ", std y: "
              << std_y << ", std theta: " << std_theta << " and weight: 1 " << std::endl;
}

void ParticleFilter::prediction(double delta_t, const double std_pos[],
                                double velocity, double yaw_rate) {
    /**
     * TODO: Add measurements to each particle and add random Gaussian noise.
     * NOTE: When adding noise you may find std::normal_distribution
     *   and std::default_random_engine useful.
     *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
     *  http://www.cplusplus.com/reference/random/default_random_engine/
     */
    std::default_random_engine gen;

    double std_x, std_y, std_theta;  // Standard deviations for x, y, and theta

    // Set standard deviations for x, y, and theta
    std_x = std_pos[0];
    std_y = std_pos[1];
    std_theta = std_pos[2];

    for (auto &particle : particles) {
        double x0 = particle.x;
        double y0 = particle.y;
        double t0 = particle.theta;

        // This line creates a normal (Gaussian) distribution for x,y,theta
        normal_distribution<double> dist_x(x0, std_x);
        normal_distribution<double> dist_y(y0, std_y);
        normal_distribution<double> dist_theta(t0, std_theta);

        // Add noise
        x0 = dist_x(gen);
        y0 = dist_y(gen);
        t0 = dist_theta(gen);

        particle.x = x0 + (velocity / yaw_rate) * (sin((t0 + delta_t * yaw_rate)) - sin(t0));
        particle.y = y0 + (velocity / yaw_rate) * (cos((t0)) - cos((t0 + delta_t * yaw_rate)));
        particle.theta = t0 + delta_t * yaw_rate;
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> &predicted,
                                     vector<LandmarkObs> &observations) {
    /**
     * TODO: Find the predicted measurement that is closest to each
     *   observed measurement and assign the observed measurement to this
     *   particular landmark.
     * NOTE: this method will NOT be called by the grading code. But you will
     *   probably find it useful to implement this method and use it as a helper
     *   during the updateWeights phase.
     */
    // Calc euclidean dist to each particle and take the closest
    for (auto pred : predicted) {
        int min_id = 0;
        double min_euclid_dist = -1;
        for (auto obs : observations) {
            double dist = sqrt(pow(pred.x - obs.x, 2) + pow(pred.y - obs.y, 2));
            if (dist < min_euclid_dist | min_euclid_dist < 0) {
                min_euclid_dist = dist;
                min_id = obs.id;
            }
        }
        pred.id = min_id;
    }

}

double multi_variant_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                          double pred_x, double pred_y) {
    // calculate normalization term
    double gauss_norm;
    gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

    // calculate exponent
    double exponent;
    exponent = (pow(x_obs - pred_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - pred_y, 2) / (2 * pow(sig_y, 2)));

    // calculate weight using normalization terms and exponent
    double weight;
    weight = gauss_norm * exp(-exponent);

    return weight;
}

vector<LandmarkObs> car2map_coordinates(const vector<LandmarkObs> &observations, double &x, double &y, double &theta) {
    vector<LandmarkObs> pos_m;
    for (auto &obs : observations) {
        LandmarkObs pos{};
        pos.x = x + obs.x + (cos(theta) * obs.x) - (sin(theta) * obs.y);
        pos.y = y + obs.y + (sin(theta) * obs.x) + (cos(theta) * obs.y);
        pos.id = obs.id;
        pos_m.push_back(pos);
    }
    return pos_m;
}

void ParticleFilter::updateWeights(double sensor_range, const double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    /**
     * TODO: Update the weights of each particle using a mult-variate Gaussian
     *   distribution. You can read more about this distribution here:
     *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
     * NOTE: The observations are given in the VEHICLE'S coordinate system.
     *   Your particles are located according to the MAP'S coordinate system.
     *   You will need to transform between the two systems. Keep in mind that
     *   this transformation requires both rotation AND translation (but no scaling).
     *   The following is a good resource for the theory:
     *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
     *   and the following is a good resource for the actual equation to implement
     *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
     */

    // clear the weights
    weights.clear();


    // Go through all particles
    for (auto &particle : particles) {


        vector<LandmarkObs> obs_map = car2map_coordinates(observations, particle.x, particle.y, particle.theta);

        // sort out nearest landmark from observations to the particle
        std::vector<LandmarkObs> landmarks_near_particle;
        for (auto &lm : map_landmarks.landmark_list) {
            double dist = sqrt(pow(particle.x - lm.x_f, 2) + pow(particle.y - lm.y_f, 2));
            if (dist <= sensor_range) {
                LandmarkObs new_lm{};
                new_lm.x = lm.x_f;
                new_lm.y = lm.y_f;
                new_lm.id = lm.id_i;
                landmarks_near_particle.push_back(new_lm);
            }
        }

        // link nearest landmark to obs by id
        dataAssociation(landmarks_near_particle, obs_map);

        particle.weight = 1;
        // update weight for found landmarks
        for (auto &obs_lm : obs_map) {
            for (auto &pred_lm : landmarks_near_particle) {
                if (pred_lm.id == obs_lm.id) {
                    particle.weight = multi_variant_prob(std_landmark[0], std_landmark[1], obs_lm.x, obs_lm.y,
                                                         pred_lm.x, pred_lm.y);
                }
            }
        }

        // save weights
        weights.push_back(particle.weight);
    }

}

void ParticleFilter::resample() {
    /**
     * TODO: Resample particles with replacement with probability proportional
     *   to their weight.
     * NOTE: You may find std::discrete_distribution helpful here.
     *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
     */
    /* Implementation of the resampling algorithm of sebastian thrun*/
    std::default_random_engine gen;
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

    vector<double> weights_res;
    for (int i = 0; i < num_particles; i++) {
        weights_res.push_back(particles[i].weight);
    }

    double max_w = *max_element(weights_res.begin(), weights_res.end());
    unsigned long N = weights_res.size();
    vector<Particle> next_step_particles;

    while (next_step_particles.size() < N) {
        unsigned long index = long(uniform_dist(gen) * (N + 1)) - 1;
        double beta = uniform_dist(gen) * 2 * max_w;
        while (weights_res[index] < beta) {
            beta = beta - weights_res[index];
            index = (index + 1) % weights_res.size();
        }
        next_step_particles.push_back(particles[index]);
    }
    particles = next_step_particles;
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}
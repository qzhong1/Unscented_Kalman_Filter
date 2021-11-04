#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 2.5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_x_;

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  weights_ = VectorXd(2 * n_aug_ + 1);

  P_ << 1, 0, 0, 0, 0,
      0, 1, 0, 0, 0,
      0, 0, 1, 0, 0,
      0, 0, 0, 0.0225, 0,
      0, 0, 0, 0, 0.0225;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  // first call initialise state x_ and covariance P_
  if (!is_initialized_) 
  {
    if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
      x_ << meas_package.raw_measurements_(0), meas_package.raw_measurements_(1), 0, 0, 0;  // random numbers for object velocity, yaw, yaw rate

    else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
      x_ << meas_package.raw_measurements_(0)*cos(meas_package.raw_measurements_(1)), meas_package.raw_measurements_(0)*sin(meas_package.raw_measurements_(1)), 0.3, 0, 0;  // random numbers for object velocity, yaw, yaw rate

    is_initialized_ = true;
  } 
  else // calls after 1st call
  { 
    time_us_ = meas_package.timestamp_;
    double dt = (time_us_ - prev_time_us_) / 1000000.0;
    prev_time_us_ = time_us_;
    Prediction(dt);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_)
      UpdateRadar(meas_package);
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_)
      UpdateLidar(meas_package);

    std::cout << "timestamp: " << time_us_ << std::endl; 
    std::cout << "state x_: " << x_.transpose() << std::endl;
  }
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */
  // create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  // create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.topRows(n_x_) = x_;
  x_aug.bottomRows(n_aug_ - n_x_).fill(0);

  // create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // create augmented covariance matrix
  MatrixXd Q = MatrixXd(n_aug_ - n_x_, n_aug_ - n_x_);
  Q << std_a_*std_a_, 0,
      0, std_yawdd_*std_yawdd_;
  P_aug.block(0,0,n_x_,n_x_) = P_;
  P_aug.block(n_x_,n_x_,n_aug_ - n_x_,n_aug_ - n_x_) = Q;

  MatrixXd A = P_aug.llt().matrixL(); // square root of P_aug

  // create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i < n_aug_; ++i) {
    Xsig_aug.col(i+1)     = x_aug + sqrt(lambda_+n_aug_) * A.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * A.col(i);
  }
  
  // predict sigma points
  for (int i=0; i<2 * n_aug_ + 1; i++){
    if(fabs(Xsig_aug(4, i)) > 0.001){
      Xsig_pred_(0,i) = Xsig_aug(0, i) + Xsig_aug(2, i)/Xsig_aug(4, i)*(sin(Xsig_aug(3, i)+Xsig_aug(4, i)*delta_t) - sin(Xsig_aug(3, i))) + 0.5*delta_t*delta_t*cos(Xsig_aug(3, i))*Xsig_aug(5, i);
      Xsig_pred_(1,i) = Xsig_aug(1, i) + Xsig_aug(2, i)/Xsig_aug(4, i)*(-cos(Xsig_aug(3, i)+Xsig_aug(4, i)*delta_t) + cos(Xsig_aug(3, i))) + 0.5*delta_t*delta_t*sin(Xsig_aug(3, i))*Xsig_aug(5, i);
      Xsig_pred_(2,i) = Xsig_aug(2, i) + delta_t*Xsig_aug(5,i);
      Xsig_pred_(3,i) = Xsig_aug(3, i) + Xsig_aug(4, i) * delta_t + 0.5*delta_t*delta_t*Xsig_aug(6,i);
      Xsig_pred_(4,i) = Xsig_aug(4, i) + delta_t*Xsig_aug(6,i);
    }else{
      Xsig_pred_(0,i) = Xsig_aug(0, i) + Xsig_aug(2, i)*cos(Xsig_aug(3, i))*delta_t + 0.5*delta_t*delta_t*cos(Xsig_aug(3, i))*Xsig_aug(5, i);
      Xsig_pred_(1,i) = Xsig_aug(1, i) + Xsig_aug(2, i)*sin(Xsig_aug(3, i))*delta_t + 0.5*delta_t*delta_t*sin(Xsig_aug(3, i))*Xsig_aug(5, i);
      Xsig_pred_(2,i) = Xsig_aug(2, i) + delta_t*Xsig_aug(5,i);
      Xsig_pred_(3,i) = Xsig_aug(3, i) + Xsig_aug(4, i) * delta_t + 0.5*delta_t*delta_t*Xsig_aug(6,i);;
      Xsig_pred_(4,i) = Xsig_aug(4, i) + delta_t*Xsig_aug(6,i);
    }
  }

  // set weights
  weights_(0) = lambda_/(lambda_+n_aug_);
  for (int i=1; i<2*n_aug_+1; i++){
    weights_(i) = 1/(2*(lambda_+n_aug_));
  }

  // predict state mean
  x_.fill(0);
  for (int i=0; i<2*n_aug_+1; i++){
    x_ += weights_(i)*Xsig_pred_.col(i);
  }


  // predict state covariance matrix
  P_.fill(0);  
  for (int i=0; i<2*n_aug_+1; i++){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;
    P_ = P_ + weights_(i) * x_diff * x_diff.transpose();
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  // set measurement dimension, lidar can measure x and y
  int n_z = 2;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  // transform sigma points into measurement space
  for (int i=0; i<2 * n_aug_ + 1; i++){
    Zsig(0, i) = Xsig_pred_(0, i);
    Zsig(1, i) = Xsig_pred_(1, i);
  }

   // calculate mean predicted measurement
  z_pred.fill(0);
  for (int i=0; i<2 * n_aug_ + 1; i++){
      z_pred += weights_(i) * Zsig.col(i);
  }
  // calculate innovation covariance matrix S
  S.fill(0);
  for (int i=0; i<2 * n_aug_ + 1; i++){
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z,n_z);
  R(0, 0) = std_laspx_*std_laspx_;
  R(1, 1) = std_laspy_*std_laspy_;
  S += R;

  // radar incoming measurement
  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0);
  for (int i = 0; i<2 * n_aug_ + 1; i++){
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = MatrixXd(n_x_, n_z);
  K = Tc*S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K*(z - z_pred);
  P_ = P_ - K*S*K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  // set measurement dimension, radar can measure r, phi, and r_dot
  int n_z = 3;

  // create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);

  // mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  // measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);

  // transform sigma points into measurement space
  for (int i=0; i<2 * n_aug_ + 1; i++){
    Zsig(0, i) = sqrt(pow(Xsig_pred_(0, i), 2) + pow(Xsig_pred_(1, i), 2));
    Zsig(1, i) = atan2(Xsig_pred_(1, i), Xsig_pred_(0, i));
    Zsig(2, i) = (Xsig_pred_(0, i)*cos(Xsig_pred_(3, i))*Xsig_pred_(2, i) + Xsig_pred_(1, i)*sin(Xsig_pred_(3, i))*Xsig_pred_(2, i))/Zsig(0, i);
  }

  // calculate mean predicted measurement
  z_pred.fill(0);
  for (int i=0; i<2 * n_aug_ + 1; i++){
      z_pred += weights_(i) * Zsig.col(i);
  }
  // calculate innovation covariance matrix S
  S.fill(0);
  for (int i=0; i<2 * n_aug_ + 1; i++){
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;
    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  MatrixXd R = MatrixXd(n_z,n_z);
  R(0, 0) = std_radr_*std_radr_;
  R(1, 1) = std_radphi_*std_radphi_;
  R(2, 2) = std_radrd_*std_radrd_;
  S += R;

  // radar incoming measurement
  VectorXd z = VectorXd(n_z);
  z = meas_package.raw_measurements_;

  // create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  // calculate cross correlation matrix
  Tc.fill(0);
  for (int i = 0; i<2 * n_aug_ + 1; i++){
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    // angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  // calculate Kalman gain K;
  MatrixXd K = MatrixXd(n_x_, n_z);
  K = Tc*S.inverse();

  // residual
  VectorXd z_diff = z - z_pred;

  // angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  // update state mean and covariance matrix
  x_ = x_ + K*(z - z_pred);
  P_ = P_ - K*S*K.transpose();
}
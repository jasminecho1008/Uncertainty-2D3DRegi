/*
 * MIT License
 *
 * Copyright (c) 2020-2022 Robert Grupp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * (License text omitted for brevity)
 */

#include <fmt/format.h>
#include <random>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

// OpenCV
#include <opencv2/imgcodecs.hpp>

// xreg and ITK includes – adjust these include paths as needed.
#include "xregCSVUtils.h"
#include "xregExceptionUtils.h"
#include "xregFilesystemUtils.h"
#include "xregHDF5.h"
#include "xregH5ProjDataIO.h"
#include "xregHUToLinAtt.h"
#include "xregITKIOUtils.h"
#include "xregITKLabelUtils.h"
#include "xregITKOpenCVUtils.h"
#include "xregITKRemapUtils.h"
#include "xregMultiObjMultiLevel2D3DRegi.h"
#include "xregMultivarNormDist.h"
#include "xregMultivarNormDistFit.h"
#include "xregNDRange.h"
#include "xregOpenCVUtils.h"
#include "xregProgOptUtils.h"
#include "xregProjPreProc.h"
#include "xregRayCastProgOpts.h"
#include "xregIntensity2D3DRegiBOBYQA.h"
#include "xregIntensity2D3DRegiExhaustive.h"
#include "xregRigidUtils.h"
#include "xregSampleUtils.h"
#include "xregSE3OptVars.h"
#include "xregStringUtils.h"

using namespace xreg;

// For brevity, we alias types. (In your code these types might be more complex.)
using FrameTransform = Eigen::Matrix4d;
using Mat4x4 = Eigen::Matrix4d;
using ProjDataF32 = struct
{
  Eigen::Matrix3d intrinsic;
  FrameTransform extrins;
  // Additional members as needed.
};
using size_type = unsigned int;

// Dummy ITK image types. In your code, these would be defined by ITK.
namespace itk
{
  template <typename T, unsigned int D>
  class Image;
  template <typename T, unsigned int D>
  using ImagePointer = typename Image<T, D>::Pointer;
  // Dummy definitions:
  template <typename T, unsigned int D>
  class Image
  {
  public:
    using Pointer = std::shared_ptr<Image>;
    static Pointer New() { return std::make_shared<Image>(); }
  };
}
using Image3D = itk::Image<float, 3>;
using Image2D = itk::Image<float, 2>;

// --------------------------------------------------------------------------
// Dummy functions for illustration – replace these with your actual implementations.
// --------------------------------------------------------------------------

// Reads a 3D CT image (in HU) from an HDF5 group.
Image3D::Pointer ReadITKImageH5Float3D(H5::Group g)
{
  std::cout << "[Dummy] Reading 3D CT volume (HU) from HDF5..." << std::endl;
  return Image3D::New();
}

// Reads a 2D image (for the projection) from an HDF5 group.
Image2D::Pointer ReadITKImageH5Float2D(H5::Group g)
{
  std::cout << "[Dummy] Reading 2D projection image from HDF5..." << std::endl;
  return Image2D::New();
}

// Converts a CT volume from HU to linear attenuation.
Image3D::Pointer HUToLinAtt(Image3D *hu)
{
  std::cout << "[Dummy] Converting HU to linear attenuation..." << std::endl;
  // In practice, apply your piecewise linear conversion.
  return Image3D::New();
}

// Simulates a DRR image from the CT linear attenuation volume, given a camera pose and projection data.
Image2D::Pointer SimulateDRR(Image3D::Pointer ct_lin_att, const FrameTransform &cam_pose, const ProjDataF32 &proj_data)
{
  std::cout << "[Dummy] Simulating DRR using full CT volume..." << std::endl;
  // Replace with your ray-casting or projection simulation.
  return Image2D::New();
}

// Writes an ITK 2D image to disk (e.g., as a NIfTI file).
void WriteITKImageToDisk(Image2D::Pointer img, const std::string &filename)
{
  std::cout << "[Dummy] Writing DRR image to " << filename << std::endl;
  // Replace with your ITK image writer code.
  std::ofstream out(filename);
  out << "Dummy DRR image data" << std::endl;
  out.close();
}

// (Optional) Create a full-volume binary mask (all ones) with the same dimensions as the input image.
Image3D::Pointer CreateFullMask(Image3D *vol)
{
  std::cout << "[Dummy] Creating full-volume mask (all ones)..." << std::endl;
  return Image3D::New();
}

// --------------------------------------------------------------------------
// SamplingToolData: holds the loaded CT, segmentation, and projection data.
struct SamplingToolData
{
  Image3D::Pointer ct_vol;  // Full CT volume (in HU)
  Image3D::Pointer seg_vol; // Segmentation mask (dummy mask if using full CT)
  ProjDataF32 pd;
  FrameTransform gt_cam_extrins_to_pelvis_vol;
  FrameTransform cam_to_pelvis_vol;
};

// Reads the CT volume, projection, and ground-truth pose from HDF5.
// MODIFICATION 1: Instead of cropping to pelvis, we use the full CT volume.
SamplingToolData ReadPelvisVolProjAndGtFromH5File(
    const std::string &h5_path,
    const std::string &spec_id_str,
    const size_type proj_idx,
    const FrameTransform &trans_m,
    std::ostream &vout)
{
  SamplingToolData data;

  vout << "-----------------------------------------\n\n";
  vout << "reading data from HDF5 file..." << std::endl;

  vout << "opening source H5 for reading: " << h5_path << std::endl;
  H5::H5File h5(h5_path, H5F_ACC_RDONLY);

  if (!ObjectInGroupH5("proj-params", h5))
  {
    xregThrow("proj-params group not found in HDF5 file!");
  }

  vout << "setting up camera..." << std::endl;
  H5::Group proj_params_g = h5.openGroup("proj-params");

  data.pd.cam.setup(ReadMatrixH5CoordScalar("intrinsic", proj_params_g),
                    ReadMatrixH5CoordScalar("extrinsic", proj_params_g),
                    ReadSingleScalarH5ULong("num-rows", proj_params_g),
                    ReadSingleScalarH5ULong("num-cols", proj_params_g),
                    ReadSingleScalarH5CoordScalar("pixel-row-spacing", proj_params_g),
                    ReadSingleScalarH5CoordScalar("pixel-col-spacing", proj_params_g));

  if (!ObjectInGroupH5(spec_id_str, h5))
  {
    xregThrow("specimen ID not found in HDF5 file: %s", spec_id_str.c_str());
  }

  H5::Group spec_g = h5.openGroup(spec_id_str);

  vout << "reading full CT intensity volume..." << std::endl;
  data.ct_vol = ReadITKImageH5Float3D(spec_g.openGroup("vol"));

  // Instead of remapping a pelvis segmentation, we create a full-volume mask.
  vout << "creating dummy full-volume mask..." << std::endl;
  data.seg_vol = CreateFullMask(data.ct_vol.GetPointer());

  H5::Group projs_g = spec_g.openGroup("projections");
  const std::string proj_idx_str = fmt::format("{:03d}", proj_idx);
  if (!ObjectInGroupH5(proj_idx_str, projs_g))
  {
    xregThrow("projection not found: %s", proj_idx_str.c_str());
  }
  H5::Group proj_g = projs_g.openGroup(proj_idx_str);

  vout << "reading projection pixels..." << std::endl;
  data.pd.img = ReadITKImageH5Float2D(proj_g.openGroup("image"));

  vout << "setting rot-up field..." << std::endl;
  data.pd.rot_to_pat_up = ReadSingleScalarH5Bool("rot-180-for-up", proj_g) ? ProjDataRotToPatUp::kONE_EIGHTY : ProjDataRotToPatUp::kZERO;

  // Read the ground-truth pose from HDF5.
  MatMxN cam_to_pelvis_vol_mat_dyn = ReadMatrixH5CoordScalar("cam-to-pelvis-vol", proj_g.openGroup("gt-poses"));
  Mat4x4 cam_to_pelvis_vol_mat = cam_to_pelvis_vol_mat_dyn;
  FrameTransform cam_to_pelvis_vol(cam_to_pelvis_vol_mat);

  // Correct for a -0.5 mm offset (as in the original code)
  {
    FrameTransform gt_corr = FrameTransform::Identity();
    gt_corr(0, 3) = -0.5;
    gt_corr(1, 3) = -0.5;
    gt_corr(2, 3) = -0.5;
    cam_to_pelvis_vol = gt_corr * cam_to_pelvis_vol;
  }

  data.cam_to_pelvis_vol = cam_to_pelvis_vol;
  data.gt_cam_extrins_to_pelvis_vol = trans_m * cam_to_pelvis_vol;

  vout << "ground truth cam extrins to pelvis vol:\n"
       << data.gt_cam_extrins_to_pelvis_vol << std::endl;
  vout << "-----------------------------------------\n\n";
  return data;
}

// --------------------------------------------------------------------------
// Pose sampling functions
// (We use the existing PoseParamSampler classes; here we assume that
// the sampler returns a 6xN matrix of pose parameter perturbations.)
// --------------------------------------------------------------------------

// For simplicity, we assume that the SE3 operator 'se3()' converts a 6-vector
// to a 4x4 rigid transform (e.g. using a Lie algebra exponential map).
// You should replace this with your actual implementation.
FrameTransform se3(const Eigen::VectorXd &params)
{
  // Here we assume the first three entries are rotation (in radians) about X, Y, Z
  // and the next three entries are translations (in mm).
  Eigen::AngleAxisd rot_x(params(0), Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd rot_y(params(1), Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd rot_z(params(2), Eigen::Vector3d::UnitZ());
  Eigen::Matrix3d R = (rot_z * rot_y * rot_x).toRotationMatrix();
  FrameTransform T = FrameTransform::Identity();
  T.block<3, 3>(0, 0) = R;
  T(0, 3) = params(3);
  T(1, 3) = params(4);
  T(2, 3) = params(5);
  return T;
}

// --------------------------------------------------------------------------
// Main function
// --------------------------------------------------------------------------
int main(int argc, char *argv[])
{
  constexpr int kEXIT_VAL_SUCCESS = 0;
  constexpr int kEXIT_VAL_BAD_USE = 1;
  constexpr int kEXIT_VAL_BAD_INPUT_HDF5 = 2;

  // Parse command-line arguments
  ProgOpts po;
  xregPROG_OPTS_SET_COMPILE_DATE(po);
  po.set_help("Generate DRRs for the full CT volume by sampling around a given ground-truth pose.");
  po.set_arg_usage("<HDF5 Data File> <patient ID> <projection index> <num samples> <output directory>");
  po.set_min_num_pos_args(5);
  po.add_backend_flags();
  try
  {
    po.parse(argc, argv);
  }
  catch (const ProgOpts::Exception &e)
  {
    std::cerr << "Error parsing command line arguments: " << e.what() << std::endl;
    po.print_usage(std::cerr);
    return kEXIT_VAL_BAD_USE;
  }
  if (po.help_set())
  {
    po.print_usage(std::cout);
    po.print_help(std::cout);
    return kEXIT_VAL_SUCCESS;
  }

  std::ostream &vout = po.vout();
  const std::string ipcai_h5_data_path = po.pos_args()[0];
  const std::string spec_id_str = po.pos_args()[1];
  const size_type proj_idx = StringCast<size_type>(po.pos_args()[2]);
  const size_type num_samples = StringCast<size_type>(po.pos_args()[3]);
  const std::string dst_dir_path = po.pos_args()[4];

  // For simplicity, we use the "mvn-approx" sampler here.
  std::string sampling_method = "mvn-approx";

  // Create output directory if necessary.
  Path dst_dir = dst_dir_path;
  if (dst_dir.exists() && !dst_dir.is_dir())
  {
    std::cerr << "ERROR: output directory path exists, but is not a directory: " << dst_dir_path << std::endl;
    return kEXIT_VAL_BAD_USE;
  }
  if (!dst_dir.exists())
  {
    vout << "creating output directory..." << std::endl;
    MakeDirRecursive(dst_dir_path);
  }

  // Seed RNG
  std::mt19937 rng_eng;
  if (po.has("rng-seed"))
  {
    rng_eng.seed(po.get("rng-seed").as_uint32());
  }
  else
  {
    SeedRNGEngWithRandDev(&rng_eng);
  }

  // --------------------------------------------------------------------------
  // Read full CT volume, projection, and ground-truth pose from HDF5.
  // MODIFICATION 1: We use the full CT volume (do not crop using segmentation).
  // --------------------------------------------------------------------------
  vout << "reading data from IPCAI HDF5 file..." << std::endl;
  SamplingToolData data_from_h5 = ReadPelvisVolProjAndGtFromH5File(
      ipcai_h5_data_path, spec_id_str, proj_idx, FrameTransform::Identity(), vout);

  // Instead of cropping to pelvis (using segmentation), use the full CT volume:
  auto ct_hu = data_from_h5.ct_vol;
  vout << "using full CT volume for DRR simulation." << std::endl;

  // Convert HU to linear attenuation (for the full CT volume)
  vout << "converting HU --> Linear Attenuation..." << std::endl;
  Image3D::Pointer ct_lin_att = HUToLinAtt(ct_hu.GetPointer());

  // --------------------------------------------------------------------------
  // Set up projection parameters.
  // (Here we assume the projection parameters in data_from_h5.pd are correct.)
  ProjDataF32 proj_data = data_from_h5.pd;

  // --------------------------------------------------------------------------
  // Set up pose sampling around the ground-truth pose.
  // The ground-truth pose (for projection index) is stored in:
  FrameTransform gt_pose = data_from_h5.gt_cam_extrins_to_pelvis_vol;
  vout << "Ground-truth pose (from HDF5):\n"
       << gt_pose << std::endl;

  // Here, we use a sampler (for example, the multivariate normal approximator)
  // that returns a 6xN matrix of pose parameter perturbations.
  // (For simplicity, we assume that the sampler’s standard deviations and grid parameters are defined in the ProgOpts.)
  // In this modified version, the sampled pose is combined with the ground-truth.
  // For each sample, we compute:
  //    sample_pose = se3(pose_params) * gt_pose;
  // where se3(pose_params) is the offset (perturbation) as a 4x4 rigid transform.
  // We then simulate a DRR using sample_pose.

  // (For this example, we use a dummy sampler that generates small random perturbations.)
  Eigen::MatrixXd pose_param_samples(6, num_samples);
  {
    std::normal_distribution<double> trans_dist(0.0, 2.0); // mm perturbation
    std::normal_distribution<double> rot_dist(0.0, 1.0);   // degree perturbation
    for (size_type s = 0; s < num_samples; ++s)
    {
      pose_param_samples(0, s) = rot_dist(rng_eng) * (M_PI / 180.0); // rotation X in rad
      pose_param_samples(1, s) = rot_dist(rng_eng) * (M_PI / 180.0); // rotation Y in rad
      pose_param_samples(2, s) = rot_dist(rng_eng) * (M_PI / 180.0); // rotation Z in rad
      pose_param_samples(3, s) = trans_dist(rng_eng);                // translation X (mm)
      pose_param_samples(4, s) = trans_dist(rng_eng);                // translation Y (mm)
      pose_param_samples(5, s) = trans_dist(rng_eng);                // translation Z (mm)
    }
  }

  // --------------------------------------------------------------------------
  // For each sample, compute the sampled pose and simulate a DRR.
  // We'll also write the 4x4 pose matrices into a CSV file.
  // --------------------------------------------------------------------------
  std::ofstream csv_file((dst_dir.string() + "/sampled_poses.csv").c_str());
  if (!csv_file.is_open())
  {
    std::cerr << "ERROR: could not open CSV file for writing." << std::endl;
    return kEXIT_VAL_BAD_USE;
  }
  // Write header for CSV (flattened 4x4 matrix)
  csv_file << "sample,";
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      csv_file << "m" << i << j;
      if (!(i == 3 && j == 3))
        csv_file << ",";
      else
        csv_file << "\n";
    }
  }

  // Loop over each sample
  for (size_type s = 0; s < num_samples; ++s)
  {
    // Compute the perturbation as a 4x4 transform.
    FrameTransform offset = se3(pose_param_samples.col(s));
    // Compose with ground-truth: sample_pose = offset * gt_pose.
    FrameTransform sample_pose = offset * gt_pose;

    // Simulate the DRR using the full CT linear attenuation volume,
    // the sampled pose, and the projection parameters.
    Image2D::Pointer drr = SimulateDRR(ct_lin_att, sample_pose, proj_data);

    // Save the DRR image (for example, as a NIfTI file).
    // (Replace WriteITKImageToDisk with your actual image writer.)
    std::string drr_filename = fmt::format("{}/drr_sample_{:03d}.png", dst_dir.string(), s);
    WriteITKImageToDisk(drr, drr_filename);
    vout << "Saved DRR for sample " << s << " to " << drr_filename << std::endl;

    // Write the sampled pose to CSV (flatten the 4x4 matrix).
    csv_file << s;
    for (int i = 0; i < 4; ++i)
    {
      for (int j = 0; j < 4; ++j)
      {
        csv_file << "," << sample_pose(i, j);
      }
    }
    csv_file << "\n";
  }
  csv_file.close();
  vout << "Finished. DRRs and pose CSV saved in " << dst_dir.string() << std::endl;

  return kEXIT_VAL_SUCCESS;
}

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
#include <opencv2/imgcodecs.hpp>

// xreg headers
#include "xregCSVUtils.h"
#include "xregEdgesFromRayCast.h" // (if needed for edge images)
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

//--------------------------------------------------------------------
// Data reading and helper function definitions
//--------------------------------------------------------------------
struct SamplingToolData
{
  itk::Image<float, 3>::Pointer ct_vol;          // Full CT volume in HU.
  itk::Image<unsigned char, 3>::Pointer seg_vol; // Segmentation volume.
  ProjDataF32 pd;                                // Projection data (intrinsics, extrinsics, projection image, etc.)
  FrameTransform gt_cam_extrins_to_pelvis_vol;   // Ground-truth camera extrinsic (4x4 matrix).
  FrameTransform cam_to_pelvis_vol;              // (Optional additional transform.)
};

SamplingToolData ReadPelvisVolProjAndGtFromH5File(
    const std::string &h5_path,
    const std::string &spec_id_str,
    const size_type proj_idx,
    const FrameTransform &trans_m,
    std::ostream &vout)
{
  SamplingToolData data;
  vout << "-----------------------------------------\n\n";
  vout << "Reading data from HDF5 file..." << std::endl;

  vout << "Opening HDF5 file: " << h5_path << std::endl;
  H5::H5File h5(h5_path, H5F_ACC_RDONLY);

  if (!ObjectInGroupH5("proj-params", h5))
    xregThrow("proj-params group not found in HDF5 file!");

  vout << "Setting up camera..." << std::endl;
  H5::Group proj_params_g = h5.openGroup("proj-params");
  data.pd.cam.setup(ReadMatrixH5CoordScalar("intrinsic", proj_params_g),
                    ReadMatrixH5CoordScalar("extrinsic", proj_params_g),
                    ReadSingleScalarH5ULong("num-rows", proj_params_g),
                    ReadSingleScalarH5ULong("num-cols", proj_params_g),
                    ReadSingleScalarH5CoordScalar("pixel-row-spacing", proj_params_g),
                    ReadSingleScalarH5CoordScalar("pixel-col-spacing", proj_params_g));

  if (!ObjectInGroupH5(spec_id_str, h5))
    xregThrow("Specimen ID not found in HDF5 file: %s", spec_id_str.c_str());

  H5::Group spec_g = h5.openGroup(spec_id_str);

  vout << "Reading full CT intensity volume (HU)..." << std::endl;
  data.ct_vol = ReadITKImageH5Float3D(spec_g.openGroup("vol"));

  vout << "Reading segmentation volume..." << std::endl;
  data.seg_vol = ReadITKImageH5UChar3D(spec_g.openGroup("vol-seg/image"));

  H5::Group projs_g = spec_g.openGroup("projections");
  const std::string proj_idx_str = fmt::format("{:03d}", proj_idx);
  if (!ObjectInGroupH5(proj_idx_str, projs_g))
    xregThrow("Projection not found: %s", proj_idx_str.c_str());
  H5::Group proj_g = projs_g.openGroup(proj_idx_str);

  vout << "Reading projection image..." << std::endl;
  data.pd.img = ReadITKImageH5Float2D(proj_g.openGroup("image"));

  vout << "Setting rot-up field..." << std::endl;
  data.pd.rot_to_pat_up = ReadSingleScalarH5Bool("rot-180-for-up", proj_g)
                              ? ProjDataRotToPatUp::kONE_EIGHTY
                              : ProjDataRotToPatUp::kZERO;

  // Read ground-truth 4x4 camera-to-volume pose.
  MatMxN cam_to_pelvis_vol_mat_dyn = ReadMatrixH5CoordScalar("cam-to-pelvis-vol", proj_g.openGroup("gt-poses"));
  Mat4x4 cam_to_pelvis_vol_mat = cam_to_pelvis_vol_mat_dyn;
  FrameTransform cam_to_pelvis_vol(cam_to_pelvis_vol_mat);
  // Correct for linear interpolation texture indexing:
  {
    FrameTransform gt_corr = FrameTransform::Identity();
    gt_corr.matrix()(0, 3) = -0.5f;
    gt_corr.matrix()(1, 3) = -0.5f;
    gt_corr.matrix()(2, 3) = -0.5f;
    cam_to_pelvis_vol = gt_corr * cam_to_pelvis_vol;
  }
  data.cam_to_pelvis_vol = cam_to_pelvis_vol;
  // Apply any additional transformation from the command line:
  data.gt_cam_extrins_to_pelvis_vol = trans_m * cam_to_pelvis_vol;

  vout << "Ground truth cam extrins to pelvis vol:\n"
       << data.gt_cam_extrins_to_pelvis_vol.matrix() << std::endl;
  vout << "-----------------------------------------\n\n";
  return data;
}

//--------------------------------------------------------------------
// Pose sampling (independent normal sampler)
class PoseParamSampler
{
public:
  virtual MatMxN SamplePoseParams(const size_type num_samples, std::mt19937 &rng_eng) = 0;
};

class PoseParamSamplerIndepNormalDims : public PoseParamSampler
{
public:
  PoseParamSamplerIndepNormalDims(const PtN &std_devs)
  {
    xregASSERT(std_devs.size() == 6);
    PtN mean(6);
    mean.setZero();
    dist_ = std::make_shared<MultivarNormalDistZeroCov>(mean, std_devs);
  }
  MatMxN SamplePoseParams(const size_type num_samples, std::mt19937 &rng_eng) override
  {
    return dist_->draw_samples(num_samples, rng_eng);
  }

private:
  std::shared_ptr<MultivarNormalDistZeroCov> dist_;
};

//--------------------------------------------------------------------
// Main: simulate DRRs by sampling around GT pose and saving CSV files.
// We generate DRRs for the whole CT volume (converted from HU to linear attenuation)
// instead of just bone.
int main(int argc, char *argv[])
{
  constexpr int kEXIT_VAL_SUCCESS = 0;
  constexpr int kEXIT_VAL_BAD_USE = 1;
  constexpr int kEXIT_VAL_BAD_INPUT_HDF5 = 2;

  // Set up program options.
  ProgOpts po;
  xregPROG_OPTS_SET_COMPILE_DATE(po);
  po.set_help("Simulate DRRs by sampling around the ground-truth pose.\n"
              "Usage: <HDF5 Data File> <patient ID> <projection index> <num samples> <output directory>");
  po.set_min_num_pos_args(5);
  po.add("method", 'm', ProgOpts::kSTORE_STRING, "method",
         "Sampling method: \"uniform\", \"prior\", or \"mvn-approx\". (Using \"uniform\" here for simplicity)")
      << "uniform";
  po.add("N", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_UINT32, "N", "Case index N (for naming output files); default 0.")
      << ProgOpts::uint32(0);
  po.add("M", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_UINT32, "M", "Case index M (for naming output files); default 0.")
      << ProgOpts::uint32(0);
  po.add("rng-seed", ProgOpts::kNO_SHORT_FLAG, ProgOpts::kSTORE_UINT32, "rng-seed", "Optional RNG seed.");
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

  // Set standard deviations for independent sampling (rotations in radians, translations in mm)
  PtN sampler_std_devs(6);
  sampler_std_devs << 0.05, 0.05, 0.05, 2.0, 2.0, 2.0;

  auto param_sampler = std::make_shared<PoseParamSamplerIndepNormalDims>(sampler_std_devs);

  // For this simulation, we use an identity transformation for extra perturbation.
  FrameTransform trans_m = EulerRotXYZTransXYZFrame(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  vout << "User-specified transformation (applied to GT):\n"
       << trans_m.matrix() << "\n";

  // Read data from HDF5.
  const auto data_from_h5 = ReadPelvisVolProjAndGtFromH5File(
      ipcai_h5_data_path, spec_id_str, proj_idx, trans_m, vout);

  // Convert the full CT volume (in HU) to linear attenuation.
  vout << "Converting full CT (HU) to linear attenuation..." << std::endl;
  auto ct_lin_att = HUToLinAtt(data_from_h5.ct_vol.GetPointer(), -1000.0f);

  // Set up the ray caster to generate DRRs.
  auto ray_caster = LineIntRayCasterFromProgOpts(po);
  ray_caster->set_ct_volume(ct_lin_att);

  // Containers to store CSV data.
  std::vector<CoordScalarList> pose_csv_rows;   // Flattened 4x4 sample poses.
  std::vector<CoordScalarList> offset_csv_rows; // Offset decompositions.
  pose_csv_rows.reserve(num_samples);
  offset_csv_rows.reserve(num_samples);

  // Seed RNG.
  std::mt19937 rng_eng;
  if (po.has("rng-seed"))
  {
    rng_eng.seed(po.get("rng-seed").as_uint32());
    vout << "Using specified RNG seed: " << po.get("rng-seed").as_uint32() << std::endl;
  }
  else
  {
    vout << "Seeding RNG with random device..." << std::endl;
    SeedRNGEngWithRandDev(&rng_eng);
  }

  // Sample pose parameters (6D: 3 rotations, 3 translations).
  MatMxN pose_param_samples = param_sampler->SamplePoseParams(num_samples, rng_eng);

  // Loop over each sample.
  for (size_type sample_idx = 0; sample_idx < num_samples; ++sample_idx)
  {
    vout << "Processing sample index: " << sample_idx << std::endl;

    // Compute the perturbation transformation from the sampled 6D parameters.
    SE3OptVarsLieAlg se3;
    FrameTransform perturb = se3(pose_param_samples.col(sample_idx));

    // Compute the simulated camera-to-volume pose by applying the perturbation to the GT.
    FrameTransform sample_pose = perturb * data_from_h5.gt_cam_extrins_to_pelvis_vol;

    // --- Save CSV data ---
    // (a) Save the flattened 4x4 transformation matrix.
    CoordScalarList tmp_pose_csv_row(16);
    int flat_idx = 0;
    for (int r = 0; r < 4; ++r)
      for (int c = 0; c < 4; ++c, ++flat_idx)
        tmp_pose_csv_row[flat_idx] = sample_pose.matrix()(r, c);
    pose_csv_rows.push_back(tmp_pose_csv_row);

    // (b) Save the 6D pose parameters that were sampled.
    // (This is optional; you may choose to write these if desired.)
    // For example:
    /*
    CoordScalarList tmp_pose_params_csv_row(6);
    for (int i = 0; i < 6; ++i)
      tmp_pose_params_csv_row[i] = pose_param_samples(i, sample_idx);
    // (Store these in a separate container if needed.)
    */

    // (c) Compute and save the offset (relative transform) decompositions.
    // Compute the relative transform: sample_pose relative to GT.
    FrameTransform relative_transform = sample_pose * data_from_h5.gt_cam_extrins_to_pelvis_vol.inverse();
    CoordScalar total_rot_rad, total_trans_mm;
    std::tie(total_rot_rad, total_trans_mm) = ComputeRotAngTransMag(relative_transform);
    CoordScalar total_rot_deg = total_rot_rad * kRAD2DEG;
    CoordScalar rot_x, rot_y, rot_z, trans_x, trans_y, trans_z;
    std::tie(rot_x, rot_y, rot_z, trans_x, trans_y, trans_z) = RigidXformToEulerXYZAndTrans(relative_transform);
    rot_x *= kRAD2DEG;
    rot_y *= kRAD2DEG;
    rot_z *= kRAD2DEG;
    CoordScalarList tmp_decomp_offset_row(8);
    tmp_decomp_offset_row[0] = total_rot_deg;
    tmp_decomp_offset_row[1] = total_trans_mm;
    tmp_decomp_offset_row[2] = rot_x;
    tmp_decomp_offset_row[3] = rot_y;
    tmp_decomp_offset_row[4] = rot_z;
    tmp_decomp_offset_row[5] = trans_x;
    tmp_decomp_offset_row[6] = trans_y;
    tmp_decomp_offset_row[7] = trans_z;
    offset_csv_rows.push_back(tmp_decomp_offset_row);

    // --- Generate DRR ---
    // Update the projection data with the new camera extrinsics.
    ProjDataF32 pd_sample = data_from_h5.pd;
    pd_sample.cam.extrins = sample_pose;
    ray_caster->set_proj_data(pd_sample);

    // Generate the DRR image (here we use projection index 0).
    auto drr_img = ray_caster->proj(0);

    // Remap DRR image to 8bpp for saving as PNG.
    cv::Mat drr_remap = ShallowCopyItkToOpenCV(ITKImageRemap8bpp(drr_img.GetPointer()).GetPointer()).clone();

    // Save the DRR images.
    const std::string sample_idx_str = fmt::format("{:03d}", sample_idx);
    std::string drr_raw_filename = fmt::format("{}/drr_raw_{}.nii.gz", dst_dir_path, sample_idx_str);
    std::string drr_png_filename = fmt::format("{}/drr_remap_{}.png", dst_dir_path, sample_idx_str);
    WriteITKImageToDisk(drr_img.GetPointer(), drr_raw_filename);
    cv::imwrite(drr_png_filename, drr_remap);
    vout << "Saved DRR (raw NIfTI): " << drr_raw_filename << std::endl;
    vout << "Saved DRR (PNG): " << drr_png_filename << std::endl;
  } // end for each sample

  // Write CSV file for the flattened pose matrices.
  std::string csv_pose_filename = fmt::format("{}/case_{}_par_{}.csv", dst_dir_path,
                                              po.get("N").as_uint32(), po.get("M").as_uint32());
  WriteCSVFile(csv_pose_filename, pose_csv_rows,
               {"m00", "m01", "m02", "m03",
                "m10", "m11", "m12", "m13",
                "m20", "m21", "m22", "m23",
                "m30", "m31", "m32", "m33"});
  vout << "Wrote pose CSV file: " << csv_pose_filename << std::endl;

  // Write CSV file for the offset decompositions.
  std::string csv_offset_filename = fmt::format("{}/case_{}_offsets.csv", dst_dir_path,
                                                po.get("N").as_uint32());
  vout << "Writing offset CSV file..." << std::endl;
  WriteCSVFile(csv_offset_filename, offset_csv_rows,
               {"total rotation (deg)", "total trans. (mm)",
                "rotation X (deg)", "rotation Y (deg)", "rotation Z (deg)",
                "translation X (mm)", "translation Y (mm)", "translation Z (mm)"});
  vout << "Wrote offset CSV file: " << csv_offset_filename << std::endl;

  vout << "Exiting..." << std::endl;
  return kEXIT_VAL_SUCCESS;
}

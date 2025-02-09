// // minimal_uniform_sample.cpp
// //
// // Minimal test application using xreg routines to:
// //  1. Read an HDF5 file containing CT, segmentation, and projection data.
// //  2. Uniformly sample a pose offset (ignoring the original ground-truth).
// //  3. Generate a DRR image via projection pre-processing.
// //  4. Extract edges using OpenCV's Canny.
// //  5. Save the DRR and edge images to disk.
// //
// // This version does no iterative registration.
// //
// // To compile, link against xreg, ITK, OpenCV, and HDF5 libraries.

// #include <fmt/format.h>
// #include <random>
// #include <iostream>
// #include <string>
// #include <stdexcept>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>

// // xreg includes (adjust include paths as needed)
// #include "xregHDF5.h"
// #include "xregH5ProjDataIO.h"
// #include "xregITKIOUtils.h"
// #include "xregProjPreProc.h"
// #include "xregITKOpenCVUtils.h" // for ShallowCopyItkToOpenCV
// #include "xregFilesystemUtils.h"
// #include "xregRigidUtils.h"
// #include "xregSE3OptVars.h"
// #include "xregSampleUtils.h"
// #include "xregStringUtils.h"
// #include "xregITKLabelUtils.h"

// using namespace xreg;

// // ---------------------------------------------------------------------------
// // Minimal structure to hold the loaded data.
// struct SamplingToolData
// {
//     itk::Image<float, 3>::Pointer ct_vol;
//     itk::Image<unsigned char, 3>::Pointer seg_vol;
//     ProjDataF32 pd;
//     FrameTransform gt_cam_extrins_to_pelvis_vol;
//     // (Other fields can be added as needed.)
// };

// // ---------------------------------------------------------------------------
// // Minimal implementation of ReadPelvisVolProjAndGtFromH5File.
// // This function uses xreg routines to open the HDF5 file and read:
// //   - The global projection parameters from group "proj-params"
// //   - The CT volume from group "<patientID>/vol"
// //   - The segmentation from group "<patientID>/vol-seg/image"
// //   - The projection data from group "<patientID>/projections/XYZ", where
// //     XYZ is a three-digit string built from proj_idx.
// //   - The ground-truth camera transform from "<patientID>/projections/XYZ/gt-poses/cam-to-pelvis-vol"
// // It also applies a fixed offset correction (as in your original code).
// SamplingToolData ReadPelvisVolProjAndGtFromH5File(
//     const std::string &h5_path,
//     const std::string &spec_id_str,
//     const size_t proj_idx,
//     const FrameTransform &trans_m,
//     std::ostream &vout)
// {
//     SamplingToolData data;

//     vout << "-----------------------------------------\n\n";
//     vout << "Reading data from HDF5 file...\n";
//     vout << "Opening source H5: " << h5_path << "\n";

//     // Open the HDF5 file.
//     H5::H5File h5(h5_path, H5F_ACC_RDONLY);

//     if (!ObjectInGroupH5("proj-params", h5))
//         xregThrow("proj-params group not found in HDF5 file!");

//     // Read global projection parameters.
//     H5::Group proj_params_g = h5.openGroup("proj-params");
//     data.pd.cam.setup(
//         ReadMatrixH5CoordScalar("intrinsic", proj_params_g),
//         ReadMatrixH5CoordScalar("extrinsic", proj_params_g),
//         ReadSingleScalarH5ULong("num-rows", proj_params_g),
//         ReadSingleScalarH5ULong("num-cols", proj_params_g),
//         ReadSingleScalarH5CoordScalar("pixel-row-spacing", proj_params_g),
//         ReadSingleScalarH5CoordScalar("pixel-col-spacing", proj_params_g));

//     // Open the specimen group.
//     if (!ObjectInGroupH5(spec_id_str, h5))
//         xregThrow("specimen ID not found in HDF5 file: %s", spec_id_str.c_str());
//     H5::Group spec_g = h5.openGroup(spec_id_str);

//     // Read the CT volume.
//     vout << "Reading intensity volume...\n";
//     data.ct_vol = ReadITKImageH5Float3D(spec_g.openGroup("vol"));

//     // Read the segmentation volume.
//     vout << "Reading segmentation volume...\n";
//     auto ct_labels = ReadITKImageH5UChar3D(spec_g.openGroup("vol-seg/image"));

//     // Remap segmentation using a simple LUT.
//     vout << "Remapping segmentation (setting label 22 to 1)...\n";
//     std::vector<unsigned char> lut(256, 0);
//     lut[22] = 1; // e.g., vertebrae_L4
//     data.seg_vol = RemapITKLabelMap<unsigned char>(ct_labels.GetPointer(), lut);

//     // Read projection data.
//     H5::Group projs_g = spec_g.openGroup("projections");
//     const std::string proj_idx_str = fmt::format("{:03d}", proj_idx);
//     if (!ObjectInGroupH5(proj_idx_str, projs_g))
//         xregThrow("projection not found: %s", proj_idx_str.c_str());
//     H5::Group proj_g = projs_g.openGroup(proj_idx_str);

//     vout << "Reading projection pixels...\n";
//     data.pd.img = ReadITKImageH5Float2D(proj_g.openGroup("image"));

//     vout << "Setting rot-up field...\n";
//     data.pd.rot_to_pat_up = ReadSingleScalarH5Bool("rot-180-for-up", proj_g)
//                                 ? ProjDataRotToPatUp::kONE_EIGHTY
//                                 : ProjDataRotToPatUp::kZERO;

//     // Read ground-truth camera pose from the projection.
//     MatMxN cam_to_pelvis_vol_mat_dyn =
//         ReadMatrixH5CoordScalar("cam-to-pelvis-vol", proj_g.openGroup("gt-poses"));
//     Mat4x4 cam_to_pelvis_vol_mat = cam_to_pelvis_vol_mat_dyn;
//     FrameTransform cam_to_pelvis_vol(cam_to_pelvis_vol_mat);

//     // Apply a fixed correction.
//     {
//         FrameTransform gt_corr = FrameTransform::Identity();
//         gt_corr.matrix()(0, 3) = -0.5f;
//         gt_corr.matrix()(1, 3) = -0.5f;
//         gt_corr.matrix()(2, 3) = -0.5f;
//         cam_to_pelvis_vol = gt_corr * cam_to_pelvis_vol;
//     }

//     // Save the camera transform.
//     data.gt_cam_extrins_to_pelvis_vol = trans_m * cam_to_pelvis_vol;
//     vout << "Ground truth cam extrins to pelvis vol:\n"
//          << data.gt_cam_extrins_to_pelvis_vol.matrix() << "\n";
//     vout << "-----------------------------------------\n\n";

//     return data;
// }

// // ---------------------------------------------------------------------------
// // Minimal main() function that uses uniform sampling to generate a DRR and edges.
// int main(int argc, char *argv[])
// {
//     if (argc < 7)
//     {
//         std::cerr << "Usage: " << argv[0]
//                   << " <HDF5 Data File> <patient ID> <projection index> <num samples> <output DRR dir> <output edges dir>\n";
//         return 1;
//     }

//     // Parse command-line arguments.
//     const std::string h5_file = argv[1];
//     const std::string patientID = argv[2];
//     const size_t proj_idx = std::stoul(argv[3]);
//     const size_t num_samples = std::stoul(argv[4]); // For minimal test, we use one sample.
//     const std::string outDRRDir = argv[5];
//     const std::string outEdgesDir = argv[6];

//     // For this minimal test, we ignore the ground-truth and use identity.
//     FrameTransform trans_m = FrameTransform::Identity();

//     std::cout << "Reading HDF5 file: " << h5_file << "\n";
//     SamplingToolData data;
//     try
//     {
//         data = ReadPelvisVolProjAndGtFromH5File(h5_file, patientID, proj_idx, trans_m, std::cout);
//     }
//     catch (const std::exception &e)
//     {
//         std::cerr << "Error reading HDF5 file: " << e.what() << "\n";
//         return 1;
//     }

//     // Uniformly sample one pose offset.
//     std::mt19937 rng(std::random_device{}());
//     // Sample rotations between -5 and 5 degrees, translations between -10 and 10 mm.
//     const double lb_rot_deg = -5.0, ub_rot_deg = 5.0;
//     const double lb_trans = -10.0, ub_trans = 10.0;
//     const double deg2rad = M_PI / 180.0;
//     double rx = std::uniform_real_distribution<double>(lb_rot_deg * deg2rad, ub_rot_deg * deg2rad)(rng);
//     double ry = std::uniform_real_distribution<double>(lb_rot_deg * deg2rad, ub_rot_deg * deg2rad)(rng);
//     double rz = std::uniform_real_distribution<double>(lb_rot_deg * deg2rad, ub_rot_deg * deg2rad)(rng);
//     double tx = std::uniform_real_distribution<double>(lb_trans, ub_trans)(rng);
//     double ty = std::uniform_real_distribution<double>(lb_trans, ub_trans)(rng);
//     double tz = std::uniform_real_distribution<double>(lb_trans, ub_trans)(rng);

//     FrameTransform pose_offset = EulerRotXYZTransXYZFrame(rx, ry, rz, tx, ty, tz);
//     std::cout << "Uniformly sampled pose offset:\n"
//               << pose_offset.matrix() << "\n";

//     // For this minimal test, ignore the ground-truth transform and use the offset alone.
//     FrameTransform new_cam_pose = pose_offset;
//     std::cout << "New camera pose (used for DRR generation):\n"
//               << new_cam_pose.matrix() << "\n";

//     // Update the projection data's camera transform.
//     data.pd.cam.extrins = new_cam_pose;

//     // Run projection pre-processing to generate a DRR image.
//     ProjPreProc proj_preproc;
//     proj_preproc.input_projs = {data.pd};
//     proj_preproc.set_debug_output_stream(std::cout, true);
//     proj_preproc();

//     // Retrieve the DRR image (an ITK image of type float, 2D).
//     itk::Image<float, 2>::Pointer drr_itk = proj_preproc.output_projs[0].img;
//     // Use xreg's function to convert from ITK to OpenCV.
//     cv::Mat drr = ShallowCopyItkToOpenCV(drr_itk.GetPointer());
//     std::cout << "DRR image generated: " << drr.rows << " x " << drr.cols << "\n";

//     // Convert DRR image to 8-bit and compute edges using Canny.
//     cv::Mat drr_8u;
//     drr.convertTo(drr_8u, CV_8U, 255.0);
//     cv::Mat edges;
//     cv::Canny(drr_8u, edges, 50, 150);

//     // Save the DRR and edge images.
//     const std::string drr_filename = outDRRDir + "/minimal_drr.png";
//     const std::string edges_filename = outEdgesDir + "/minimal_edges.png";
//     cv::imwrite(drr_filename, drr_8u);
//     cv::imwrite(edges_filename, edges);

//     std::cout << "Saved DRR image to " << drr_filename << "\n";
//     std::cout << "Saved edge image to " << edges_filename << "\n";

//     return 0;
// }

// // minimal_uniform_sample.cpp
// //
// // Minimal test application using xreg routines to:
// //  1. Read an HDF5 file containing CT, segmentation, and projection data.
// //  2. Remap the segmentation (label 22 → 1) and mask the CT volume so that only
// //     the labeled region is used; then convert HU to linear attenuation.
// //  3. Uniformly sample a pose offset (ignoring the original ground‐truth)
// //     and use that as the new camera pose.
// //  4. Run projection pre‑processing to generate a DRR image.
// //  5. Extract edges using OpenCV's Canny.
// //  6. Save the DRR and edge images to disk.
// //
// // This version does no iterative registration.
// //
// // To compile, link against xreg, ITK, OpenCV, and HDF5 libraries.

// #include <fmt/format.h>
// #include <random>
// #include <iostream>
// #include <string>
// #include <stdexcept>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>

// // xreg includes (adjust include paths as needed)
// #include "xregHDF5.h"
// #include "xregH5ProjDataIO.h"
// #include "xregITKIOUtils.h"
// #include "xregProjPreProc.h"
// #include "xregITKOpenCVUtils.h" // for xreg::ShallowCopyItkToOpenCV()
// #include "xregFilesystemUtils.h"
// #include "xregRigidUtils.h"
// #include "xregSE3OptVars.h"
// #include "xregSampleUtils.h"
// #include "xregStringUtils.h"
// #include "xregITKLabelUtils.h" // For MakeVolListFromVolAndLabels
// #include "xregHUToLinAtt.h"    // For HUToLinAtt

// using namespace xreg;

// // ---------------------------------------------------------------------------
// // Minimal structure to hold the loaded data.
// struct SamplingToolData
// {
//     itk::Image<float, 3>::Pointer ct_vol;
//     itk::Image<unsigned char, 3>::Pointer seg_vol;
//     ProjDataF32 pd;
//     FrameTransform gt_cam_extrins_to_pelvis_vol;
//     // (Other fields can be added as needed.)
// };

// // ---------------------------------------------------------------------------
// // Minimal implementation of ReadPelvisVolProjAndGtFromH5File.
// // This function uses xreg routines to open the HDF5 file and read:
// //   - Global projection parameters from group "proj-params".
// //   - The CT volume from group "<spec_id>/vol".
// //   - The segmentation from group "<spec_id>/vol-seg/image".
// //   - The projection image from group "<spec_id>/projections/<proj_idx>/".
// //   - The ground-truth camera pose from "<spec_id>/projections/<proj_idx>/gt-poses/cam-to-pelvis-vol".
// // It also applies a fixed correction.
// SamplingToolData ReadPelvisVolProjAndGtFromH5File(
//     const std::string &h5_path,
//     const std::string &spec_id_str,
//     const size_t proj_idx,
//     const FrameTransform &trans_m,
//     std::ostream &vout)
// {
//     SamplingToolData data;
//     vout << "-----------------------------------------\n\n";
//     vout << "Reading data from HDF5 file...\n";
//     vout << "Opening source H5 for reading: " << h5_path << "\n";

//     H5::H5File h5(h5_path, H5F_ACC_RDONLY);
//     if (!ObjectInGroupH5("proj-params", h5))
//         xregThrow("proj-params group not found in HDF5 file!");

//     // Read global projection parameters.
//     H5::Group proj_params_g = h5.openGroup("proj-params");
//     data.pd.cam.setup(
//         ReadMatrixH5CoordScalar("intrinsic", proj_params_g),
//         ReadMatrixH5CoordScalar("extrinsic", proj_params_g),
//         ReadSingleScalarH5ULong("num-rows", proj_params_g),
//         ReadSingleScalarH5ULong("num-cols", proj_params_g),
//         ReadSingleScalarH5CoordScalar("pixel-row-spacing", proj_params_g),
//         ReadSingleScalarH5CoordScalar("pixel-col-spacing", proj_params_g));

//     if (!ObjectInGroupH5(spec_id_str, h5))
//         xregThrow("specimen ID not found in HDF5 file: %s", spec_id_str.c_str());
//     H5::Group spec_g = h5.openGroup(spec_id_str);

//     // Read CT volume.
//     vout << "Reading intensity volume...\n";
//     data.ct_vol = ReadITKImageH5Float3D(spec_g.openGroup("vol"));

//     // Read segmentation volume.
//     vout << "Reading segmentation volume...\n";
//     auto ct_labels = ReadITKImageH5UChar3D(spec_g.openGroup("vol-seg/image"));

//     vout << "Remapping segmentation (setting label 22 to 1)...\n";
//     std::vector<unsigned char> lut(256, 0);
//     lut[22] = 1; // remap label 22 to 1
//     data.seg_vol = RemapITKLabelMap<unsigned char>(ct_labels.GetPointer(), lut);

//     // Read projection data.
//     H5::Group projs_g = spec_g.openGroup("projections");
//     const std::string proj_idx_str = fmt::format("{:03d}", proj_idx);
//     if (!ObjectInGroupH5(proj_idx_str, projs_g))
//         xregThrow("projection not found: %s", proj_idx_str.c_str());
//     H5::Group proj_g = projs_g.openGroup(proj_idx_str);

//     vout << "Reading projection pixels...\n";
//     data.pd.img = ReadITKImageH5Float2D(proj_g.openGroup("image"));

//     vout << "Setting rot-up field...\n";
//     data.pd.rot_to_pat_up = ReadSingleScalarH5Bool("rot-180-for-up", proj_g)
//                                 ? ProjDataRotToPatUp::kONE_EIGHTY
//                                 : ProjDataRotToPatUp::kZERO;

//     // Read ground-truth camera pose.
//     MatMxN cam_to_pelvis_vol_mat_dyn =
//         ReadMatrixH5CoordScalar("cam-to-pelvis-vol", proj_g.openGroup("gt-poses"));
//     Mat4x4 cam_to_pelvis_vol_mat = cam_to_pelvis_vol_mat_dyn;
//     FrameTransform cam_to_pelvis_vol(cam_to_pelvis_vol_mat);

//     // Apply a fixed correction.
//     {
//         FrameTransform gt_corr = FrameTransform::Identity();
//         gt_corr.matrix()(0, 3) = -0.5f;
//         gt_corr.matrix()(1, 3) = -0.5f;
//         gt_corr.matrix()(2, 3) = -0.5f;
//         cam_to_pelvis_vol = gt_corr * cam_to_pelvis_vol;
//     }

//     data.gt_cam_extrins_to_pelvis_vol = trans_m * cam_to_pelvis_vol;
//     vout << "Ground truth cam extrins to pelvis vol:\n"
//          << data.gt_cam_extrins_to_pelvis_vol.matrix() << "\n";
//     vout << "-----------------------------------------\n\n";

//     return data;
// }

// // ---------------------------------------------------------------------------
// // Minimal main() function.
// int main(int argc, char *argv[])
// {
//     if (argc < 7)
//     {
//         std::cerr << "Usage: " << argv[0]
//                   << " <HDF5 Data File> <patient ID> <projection index> <num samples> <output DRR dir> <output edges dir>\n";
//         return 1;
//     }

//     // Parse command-line arguments.
//     const std::string h5_file = argv[1];
//     const std::string patientID = argv[2];
//     const size_t proj_idx = std::stoul(argv[3]);
//     const size_t num_samples = std::stoul(argv[4]); // For minimal test, we use one sample.
//     const std::string outDRRDir = argv[5];
//     const std::string outEdgesDir = argv[6];

//     // For this minimal test, use identity as the transformation.
//     FrameTransform trans_m = FrameTransform::Identity();

//     std::cout << "Reading HDF5 file: " << h5_file << "\n";
//     SamplingToolData data;
//     try
//     {
//         data = ReadPelvisVolProjAndGtFromH5File(h5_file, patientID, proj_idx, trans_m, std::cout);
//     }
//     catch (const std::exception &e)
//     {
//         std::cerr << "Error reading HDF5 file: " << e.what() << "\n";
//         return 1;
//     }

//     // Uniformly sample one pose offset.
//     std::mt19937 rng(std::random_device{}());
//     const double lb_rot_deg = -5.0, ub_rot_deg = 5.0;
//     const double lb_trans = -10.0, ub_trans = 10.0;
//     const double deg2rad = M_PI / 180.0;
//     double rx = std::uniform_real_distribution<double>(lb_rot_deg * deg2rad, ub_rot_deg * deg2rad)(rng);
//     double ry = std::uniform_real_distribution<double>(lb_rot_deg * deg2rad, ub_rot_deg * deg2rad)(rng);
//     double rz = std::uniform_real_distribution<double>(lb_rot_deg * deg2rad, ub_rot_deg * deg2rad)(rng);
//     double tx = std::uniform_real_distribution<double>(lb_trans, ub_trans)(rng);
//     double ty = std::uniform_real_distribution<double>(lb_trans, ub_trans)(rng);
//     double tz = std::uniform_real_distribution<double>(lb_trans, ub_trans)(rng);

//     FrameTransform pose_offset = EulerRotXYZTransXYZFrame(rx, ry, rz, tx, ty, tz);
//     std::cout << "Uniformly sampled pose offset:\n"
//               << pose_offset.matrix() << "\n";

//     // For this minimal test, use the offset as the new camera pose.
//     FrameTransform new_cam_pose = pose_offset;
//     std::cout << "New camera pose (used for DRR generation):\n"
//               << new_cam_pose.matrix() << "\n";

//     // Update the projection data's camera transform.
//     data.pd.cam.extrins = new_cam_pose;

//     // --- Masking Step ---
//     std::cout << "Masking CT volume using segmentation...\n";
//     auto ct_hu = MakeVolListFromVolAndLabels(data.ct_vol.GetPointer(),
//                                              data.seg_vol.GetPointer(),
//                                              {static_cast<unsigned char>(1)},
//                                              -1000.0f)[0];
//     // Convert HU to linear attenuation.
//     auto ct_lin_att = xreg::HUToLinAtt(ct_hu.GetPointer());

//     // Replace the CT volume with the masked, linear attenuation volume.
//     data.ct_vol = ct_lin_att;
//     // --- End Masking Step ---

//     // Run projection pre-processing (generates a DRR).
//     ProjPreProc proj_preproc;
//     proj_preproc.input_projs = {data.pd};
//     proj_preproc.set_debug_output_stream(std::cout, true);
//     proj_preproc();

//     // Retrieve the DRR image (an ITK image of type float, 2D) and convert it to an OpenCV Mat.
//     itk::Image<float, 2>::Pointer drr_itk = proj_preproc.output_projs[0].img;
//     cv::Mat drr = xreg::ShallowCopyItkToOpenCV(drr_itk.GetPointer());
//     std::cout << "DRR image generated: " << drr.rows << " x " << drr.cols << "\n";

//     // Use OpenCV Canny to extract edges.
//     cv::Mat drr_8u;
//     drr.convertTo(drr_8u, CV_8U, 255.0);
//     cv::Mat edges;
//     cv::Canny(drr_8u, edges, 50, 150);

//     // Save the DRR and edge images.
//     const std::string drr_filename = outDRRDir + "/minimal_drr.png";
//     const std::string edges_filename = outEdgesDir + "/minimal_edges.png";
//     cv::imwrite(drr_filename, drr_8u);
//     cv::imwrite(edges_filename, edges);

//     std::cout << "Saved DRR image to " << drr_filename << "\n";
//     std::cout << "Saved edge image to " << edges_filename << "\n";

//     return 0;
// }

// // minimal_uniform_sample_with_resample.cpp
// //
// // Minimal test application using xreg routines to:
// //  1. Read an HDF5 file containing CT, segmentation, and projection data.
// //  2. Resample the segmentation to match the CT volume.
// //  3. Explicitly create a binary mask from the segmentation (where segmentation==1)
// //     and use that to mask the CT volume (setting voxels outside the mask to –1000).
// //  4. Convert the masked CT volume from HU to linear attenuation.
// //  5. Uniformly sample a pose offset (ignoring the original ground‐truth)
// //     and use that as the new camera pose.
// //  6. Run projection pre‑processing to generate a DRR image.
// //  7. Extract edges using OpenCV’s Canny.
// //  8. Save the DRR and edge images to disk.
// //
// // To compile, link against xreg, ITK, OpenCV, and HDF5 libraries.

// #include <fmt/format.h>
// #include <random>
// #include <iostream>
// #include <string>
// #include <stdexcept>
// #include <opencv2/imgcodecs.hpp>
// #include <opencv2/imgproc.hpp>

// // ITK includes for reading/resampling/masking:
// #include "itkImage.h"
// #include "itkImageFileReader.h"
// #include "itkResampleImageFilter.h"
// #include "itkLinearInterpolateImageFunction.h"
// #include "itkBinaryThresholdImageFilter.h"
// #include "itkMaskImageFilter.h"
// #include "itkImageFileWriter.h"

// // xreg includes (adjust include paths as needed)
// #include "xregHDF5.h"
// #include "xregH5ProjDataIO.h"
// #include "xregITKIOUtils.h"
// #include "xregProjPreProc.h"
// #include "xregITKOpenCVUtils.h" // for ShallowCopyItkToOpenCV()
// #include "xregFilesystemUtils.h"
// #include "xregRigidUtils.h"
// #include "xregSE3OptVars.h"
// #include "xregSampleUtils.h"
// #include "xregStringUtils.h"
// #include "xregITKLabelUtils.h" // For RemapITKLabelMap
// #include "xregHUToLinAtt.h"    // For HUToLinAtt

// using namespace xreg;

// // Define image types.
// using CTImageType = itk::Image<float, 3>;
// using SegImageType = itk::Image<unsigned char, 3>;

// // ---------------------------------------------------------------------------
// // Minimal structure to hold loaded data.
// struct SamplingToolData
// {
//     CTImageType::Pointer ct_vol;
//     SegImageType::Pointer seg_vol;
//     ProjDataF32 pd;
//     FrameTransform gt_cam_extrins_to_pelvis_vol;
//     // (Additional fields can be added as needed.)
// };

// // ---------------------------------------------------------------------------
// // Minimal HDF5 reader function (simplified version).
// SamplingToolData ReadPelvisVolProjAndGtFromH5File(
//     const std::string &h5_path,
//     const std::string &spec_id_str,
//     const size_t proj_idx,
//     const FrameTransform &trans_m,
//     std::ostream &vout)
// {
//     SamplingToolData data;

//     vout << "-----------------------------------------\n\n";
//     vout << "Reading data from HDF5 file...\n";
//     vout << "Opening HDF5 file: " << h5_path << "\n";

//     H5::H5File h5(h5_path, H5F_ACC_RDONLY);
//     if (!ObjectInGroupH5("proj-params", h5))
//         xregThrow("proj-params group not found in HDF5 file!");

//     // Read global projection parameters.
//     H5::Group proj_params_g = h5.openGroup("proj-params");
//     data.pd.cam.setup(
//         ReadMatrixH5CoordScalar("intrinsic", proj_params_g),
//         ReadMatrixH5CoordScalar("extrinsic", proj_params_g),
//         ReadSingleScalarH5ULong("num-rows", proj_params_g),
//         ReadSingleScalarH5ULong("num-cols", proj_params_g),
//         ReadSingleScalarH5CoordScalar("pixel-row-spacing", proj_params_g),
//         ReadSingleScalarH5CoordScalar("pixel-col-spacing", proj_params_g));

//     if (!ObjectInGroupH5(spec_id_str, h5))
//         xregThrow("specimen ID not found in HDF5 file: %s", spec_id_str.c_str());
//     H5::Group spec_g = h5.openGroup(spec_id_str);

//     vout << "Reading intensity volume...\n";
//     data.ct_vol = ReadITKImageH5Float3D(spec_g.openGroup("vol"));

//     vout << "Reading segmentation volume...\n";
//     data.seg_vol = ReadITKImageH5UChar3D(spec_g.openGroup("vol-seg/image"));

//     vout << "Remapping segmentation (setting label 22 to 1)...\n";
//     std::vector<unsigned char> lut(256, 0);
//     lut[22] = 1; // remap label 22 to 1
//     data.seg_vol = RemapITKLabelMap<unsigned char>(data.seg_vol.GetPointer(), lut);

//     // Read projection data.
//     H5::Group projs_g = spec_g.openGroup("projections");
//     const std::string proj_idx_str = fmt::format("{:03d}", proj_idx);
//     if (!ObjectInGroupH5(proj_idx_str, projs_g))
//         xregThrow("projection not found: %s", proj_idx_str.c_str());
//     H5::Group proj_g = projs_g.openGroup(proj_idx_str);

//     vout << "Reading projection image...\n";
//     data.pd.img = ReadITKImageH5Float2D(proj_g.openGroup("image"));

//     vout << "Setting rot-up field...\n";
//     data.pd.rot_to_pat_up = ReadSingleScalarH5Bool("rot-180-for-up", proj_g)
//                                 ? ProjDataRotToPatUp::kONE_EIGHTY
//                                 : ProjDataRotToPatUp::kZERO;

//     // Read ground-truth camera pose.
//     MatMxN cam_to_pelvis_vol_mat_dyn =
//         ReadMatrixH5CoordScalar("cam-to-pelvis-vol", proj_g.openGroup("gt-poses"));
//     Mat4x4 cam_to_pelvis_vol_mat = cam_to_pelvis_vol_mat_dyn;
//     FrameTransform cam_to_pelvis_vol(cam_to_pelvis_vol_mat);

//     // Apply a fixed correction.
//     {
//         FrameTransform gt_corr = FrameTransform::Identity();
//         gt_corr.matrix()(0, 3) = -0.5f;
//         gt_corr.matrix()(1, 3) = -0.5f;
//         gt_corr.matrix()(2, 3) = -0.5f;
//         cam_to_pelvis_vol = gt_corr * cam_to_pelvis_vol;
//     }

//     data.gt_cam_extrins_to_pelvis_vol = trans_m * cam_to_pelvis_vol;
//     vout << "Ground truth camera transform:\n"
//          << data.gt_cam_extrins_to_pelvis_vol.matrix() << "\n";
//     vout << "-----------------------------------------\n\n";

//     return data;
// }

// // ---------------------------------------------------------------------------
// // Minimal main() function.
// int main(int argc, char *argv[])
// {
//     if (argc < 7)
//     {
//         std::cerr << "Usage: " << argv[0]
//                   << " <HDF5 Data File> <patient ID> <projection index> <num samples> <output DRR dir> <output edges dir>\n";
//         return 1;
//     }

//     // Parse command-line arguments.
//     const std::string h5_file = argv[1];
//     const std::string patientID = argv[2];
//     const size_t proj_idx = std::stoul(argv[3]);
//     const size_t num_samples = std::stoul(argv[4]); // For minimal test, typically 1.
//     const std::string outDRRDir = argv[5];
//     const std::string outEdgesDir = argv[6];

//     // Use identity for the transformation.
//     FrameTransform trans_m = FrameTransform::Identity();

//     std::cout << "Reading HDF5 file: " << h5_file << "\n";
//     SamplingToolData data_from_h5;
//     try
//     {
//         data_from_h5 = ReadPelvisVolProjAndGtFromH5File(h5_file, patientID, proj_idx, trans_m, std::cout);
//     }
//     catch (const std::exception &e)
//     {
//         std::cerr << "Error reading HDF5 file: " << e.what() << "\n";
//         return 1;
//     }

//     // Uniformly sample one pose offset.
//     std::mt19937 rng(std::random_device{}());
//     const double lb_rot_deg = -5.0, ub_rot_deg = 5.0;
//     const double lb_trans = -10.0, ub_trans = 10.0;
//     const double deg2rad = M_PI / 180.0;
//     double rx = std::uniform_real_distribution<double>(lb_rot_deg * deg2rad, ub_rot_deg * deg2rad)(rng);
//     double ry = std::uniform_real_distribution<double>(lb_rot_deg * deg2rad, ub_rot_deg * deg2rad)(rng);
//     double rz = std::uniform_real_distribution<double>(lb_rot_deg * deg2rad, ub_rot_deg * deg2rad)(rng);
//     double tx = std::uniform_real_distribution<double>(lb_trans, ub_trans)(rng);
//     double ty = std::uniform_real_distribution<double>(lb_trans, ub_trans)(rng);
//     double tz = std::uniform_real_distribution<double>(lb_trans, ub_trans)(rng);

//     FrameTransform pose_offset = EulerRotXYZTransXYZFrame(rx, ry, rz, tx, ty, tz);
//     std::cout << "Uniformly sampled pose offset:\n"
//               << pose_offset.matrix() << "\n";

//     // Use the offset as the new camera pose.
//     FrameTransform new_cam_pose = pose_offset;
//     std::cout << "New camera pose (for DRR generation):\n"
//               << new_cam_pose.matrix() << "\n";

//     // Update the projection data's camera transform.
//     data_from_h5.pd.cam.extrins = new_cam_pose;

//     // --- Explicit Binary Masking with Resampling ---
//     std::cout << "Masking CT volume using segmentation...\n";
//     // If the segmentation is not in the same space as the CT, resample it.
//     using ResampleFilterType = itk::ResampleImageFilter<SegImageType, SegImageType>;
//     ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
//     resampleFilter->SetInput(data_from_h5.seg_vol);
//     resampleFilter->SetOutputOrigin(data_from_h5.ct_vol->GetOrigin());
//     resampleFilter->SetOutputSpacing(data_from_h5.ct_vol->GetSpacing());
//     resampleFilter->SetOutputDirection(data_from_h5.ct_vol->GetDirection());
//     resampleFilter->SetSize(data_from_h5.ct_vol->GetLargestPossibleRegion().GetSize());
//     using LinearInterpolatorType = itk::LinearInterpolateImageFunction<SegImageType, double>;
//     LinearInterpolatorType::Pointer interpolator = LinearInterpolatorType::New();
//     resampleFilter->SetInterpolator(interpolator);
//     resampleFilter->Update();
//     // Now resampled segmentation is in the same grid as CT.
//     SegImageType::Pointer segResampled = resampleFilter->GetOutput();

//     // Create a binary mask from the resampled segmentation (keep voxels == 1).
//     using BinaryThresholdFilterType = itk::BinaryThresholdImageFilter<SegImageType, SegImageType>;
//     BinaryThresholdFilterType::Pointer binaryFilter = BinaryThresholdFilterType::New();
//     binaryFilter->SetInput(segResampled);
//     binaryFilter->SetLowerThreshold(1);
//     binaryFilter->SetUpperThreshold(1);
//     binaryFilter->SetInsideValue(1);
//     binaryFilter->SetOutsideValue(0);
//     binaryFilter->Update();

//     // Apply the binary mask to the CT volume.
//     using MaskFilterType = itk::MaskImageFilter<CTImageType, SegImageType, CTImageType>;
//     MaskFilterType::Pointer maskFilter = MaskFilterType::New();
//     maskFilter->SetInput(data_from_h5.ct_vol);
//     maskFilter->SetMaskImage(binaryFilter->GetOutput());
//     maskFilter->SetOutsideValue(-1000.0f);
//     maskFilter->Update();

//     CTImageType::Pointer maskedCT = maskFilter->GetOutput();

//     // Convert the masked CT image from HU to linear attenuation.
//     CTImageType::Pointer ct_lin_att = xreg::HUToLinAtt(maskedCT.GetPointer());

//     // Update the CT volume used for DRR generation.
//     data_from_h5.ct_vol = ct_lin_att;
//     // --- End Masking Step ---

//     // Run projection pre‑processing to generate a DRR.
//     ProjPreProc proj_preproc;
//     proj_preproc.input_projs = {data_from_h5.pd};
//     proj_preproc.set_debug_output_stream(std::cout, true);
//     proj_preproc();

//     itk::Image<float, 2>::Pointer drr_itk = proj_preproc.output_projs[0].img;
//     cv::Mat drr = xreg::ShallowCopyItkToOpenCV(drr_itk.GetPointer());
//     std::cout << "DRR image generated: " << drr.rows << " x " << drr.cols << "\n";

//     // Extract edges using OpenCV's Canny.
//     cv::Mat drr_8u;
//     drr.convertTo(drr_8u, CV_8U, 255.0);
//     cv::Mat edges;
//     cv::Canny(drr_8u, edges, 50, 150);

//     // Save the DRR and edge images.
//     const std::string drr_filename = outDRRDir + "/minimal_drr.png";
//     const std::string edges_filename = outEdgesDir + "/minimal_edges.png";
//     cv::imwrite(drr_filename, drr_8u);
//     cv::imwrite(edges_filename, edges);

//     std::cout << "Saved DRR image to " << drr_filename << "\n";
//     std::cout << "Saved edge image to " << edges_filename << "\n";

//     return 0;
// }

// xreg_minimal_test_main.cpp
//
// Minimal test application using xreg routines to:
//  1. Read an HDF5 file containing CT, segmentation, and projection data.
//  2. Resample the segmentation image (nearest-neighbor) to the CT grid.
//  3. Remap the segmentation so that (if you comment out the LUT assignment) no voxel is set to 1.
//  4. Create a binary mask from the remapped segmentation (threshold = 1).
//  5. Mask the CT volume using that binary mask (set voxels outside the mask to –1000 HU).
//  6. Convert the masked CT volume from HU to linear attenuation.
//  7. Uniformly sample a pose offset and use that as the new camera pose.
//  8. Run projection pre‑processing to generate a DRR.
//  9. Extract edges using OpenCV’s Canny.
// 10. Write out the intermediate images and the final DRR and edge images for inspection.
//
// To compile, link against xreg, ITK, OpenCV, and HDF5 libraries.

#include <fmt/format.h>
#include <random>
#include <iostream>
#include <string>
#include <stdexcept>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// ITK includes for resampling, thresholding, masking, and writing images.
#include "itkImage.h"
#include "itkResampleImageFilter.h"
#include "itkNearestNeighborInterpolateImageFunction.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkMaskImageFilter.h"
#include "itkImageFileWriter.h"

// xreg includes
#include "xregHDF5.h"
#include "xregH5ProjDataIO.h"
#include "xregITKIOUtils.h"
#include "xregProjPreProc.h"
#include "xregITKOpenCVUtils.h" // for ShallowCopyItkToOpenCV()
#include "xregFilesystemUtils.h"
#include "xregRigidUtils.h"
#include "xregSE3OptVars.h"
#include "xregSampleUtils.h"
#include "xregStringUtils.h"
#include "xregITKLabelUtils.h" // For RemapITKLabelMap
#include "xregHUToLinAtt.h"    // For HUToLinAtt

using namespace xreg;

// Define ITK image types.
using CTImageType = itk::Image<float, 3>;
using SegImageType = itk::Image<unsigned char, 3>;

// ---------------------------------------------------------------------------
// Structure to hold loaded data.
struct SamplingToolData
{
    CTImageType::Pointer ct_vol;
    SegImageType::Pointer seg_vol;
    ProjDataF32 pd;
    FrameTransform gt_cam_extrins_to_pelvis_vol;
};

// ---------------------------------------------------------------------------
// Minimal HDF5 reader function.
SamplingToolData ReadPelvisVolProjAndGtFromH5File(
    const std::string &h5_path,
    const std::string &spec_id_str,
    const size_t proj_idx,
    const FrameTransform &trans_m,
    std::ostream &vout)
{
    SamplingToolData data;
    vout << "-----------------------------------------\n\n";
    vout << "Reading data from HDF5 file...\n";
    vout << "Opening HDF5 file: " << h5_path << "\n";

    H5::H5File h5(h5_path, H5F_ACC_RDONLY);
    if (!ObjectInGroupH5("proj-params", h5))
        xregThrow("proj-params group not found in HDF5 file!");

    // Read global projection parameters.
    H5::Group proj_params_g = h5.openGroup("proj-params");
    data.pd.cam.setup(
        ReadMatrixH5CoordScalar("intrinsic", proj_params_g),
        ReadMatrixH5CoordScalar("extrinsic", proj_params_g),
        ReadSingleScalarH5ULong("num-rows", proj_params_g),
        ReadSingleScalarH5ULong("num-cols", proj_params_g),
        ReadSingleScalarH5CoordScalar("pixel-row-spacing", proj_params_g),
        ReadSingleScalarH5CoordScalar("pixel-col-spacing", proj_params_g));

    if (!ObjectInGroupH5(spec_id_str, h5))
        xregThrow("specimen ID not found in HDF5 file: %s", spec_id_str.c_str());
    H5::Group spec_g = h5.openGroup(spec_id_str);

    vout << "Reading intensity volume...\n";
    data.ct_vol = ReadITKImageH5Float3D(spec_g.openGroup("vol"));

    vout << "Reading segmentation volume...\n";
    data.seg_vol = ReadITKImageH5UChar3D(spec_g.openGroup("vol-seg/image"));

    vout << "Remapping segmentation (if LUT assignment is commented out, result should be all 0)...\n";
    {
        std::vector<unsigned char> lut(256, 0);
        // Uncomment the next line to set label 22 to 1:
        // lut[22] = 1;
        data.seg_vol = RemapITKLabelMap<unsigned char>(data.seg_vol.GetPointer(), lut);
    }

    H5::Group projs_g = spec_g.openGroup("projections");
    const std::string proj_idx_str = fmt::format("{:03d}", proj_idx);
    if (!ObjectInGroupH5(proj_idx_str, projs_g))
        xregThrow("projection not found: %s", proj_idx_str.c_str());
    H5::Group proj_g = projs_g.openGroup(proj_idx_str);

    vout << "Reading projection image...\n";
    data.pd.img = ReadITKImageH5Float2D(proj_g.openGroup("image"));

    vout << "Setting rot-up field...\n";
    data.pd.rot_to_pat_up = ReadSingleScalarH5Bool("rot-180-for-up", proj_g)
                                ? ProjDataRotToPatUp::kONE_EIGHTY
                                : ProjDataRotToPatUp::kZERO;

    MatMxN cam_to_pelvis_vol_mat_dyn =
        ReadMatrixH5CoordScalar("cam-to-pelvis-vol", proj_g.openGroup("gt-poses"));
    Mat4x4 cam_to_pelvis_vol_mat = cam_to_pelvis_vol_mat_dyn;
    FrameTransform cam_to_pelvis_vol(cam_to_pelvis_vol_mat);

    data.gt_cam_extrins_to_pelvis_vol = trans_m * cam_to_pelvis_vol;
    vout << "Ground truth camera transform:\n"
         << data.gt_cam_extrins_to_pelvis_vol.matrix() << "\n";
    vout << "-----------------------------------------\n\n";

    return data;
}

// ---------------------------------------------------------------------------
// Main function.
int main(int argc, char *argv[])
{
    if (argc < 7)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <HDF5 Data File> <patient ID> <projection index> <num samples> <output DRR dir> <output edges dir>\n";
        return 1;
    }

    // Parse command-line arguments.
    const std::string h5_file = argv[1];
    const std::string patientID = argv[2];
    const size_t proj_idx = std::stoul(argv[3]);
    const size_t num_samples = std::stoul(argv[4]); // Typically 1 for minimal testing.
    const std::string outDRRDir = argv[5];
    const std::string outEdgesDir = argv[6];

    // For minimal testing, use identity as the transformation.
    FrameTransform trans_m = FrameTransform::Identity();

    std::cout << "Reading HDF5 file: " << h5_file << "\n";
    SamplingToolData data_from_h5;
    try
    {
        data_from_h5 = ReadPelvisVolProjAndGtFromH5File(h5_file, patientID, proj_idx, trans_m, std::cout);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error reading HDF5 file: " << e.what() << "\n";
        return 1;
    }

    // --- Write out the remapped segmentation for inspection ---
    {
        using WriterType = itk::ImageFileWriter<SegImageType>;
        WriterType::Pointer writer = WriterType::New();
        std::string segFilename = outDRRDir + "/remapped_segmentation.nii.gz";
        writer->SetFileName(segFilename);
        writer->SetInput(data_from_h5.seg_vol);
        try
        {
            writer->Update();
        }
        catch (itk::ExceptionObject &ex)
        {
            std::cerr << "Error writing remapped segmentation: " << ex << "\n";
        }
        std::cout << "Wrote remapped segmentation to " << segFilename << "\n";
    }

    // --- Masking Step ---
    std::cout << "Masking CT volume using segmentation...\n";

    // Resample the segmentation to the CT grid using nearest-neighbor interpolation.
    using ResampleFilterType = itk::ResampleImageFilter<SegImageType, SegImageType>;
    ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
    resampleFilter->SetInput(data_from_h5.seg_vol);
    resampleFilter->SetOutputOrigin(data_from_h5.ct_vol->GetOrigin());
    resampleFilter->SetOutputSpacing(data_from_h5.ct_vol->GetSpacing());
    resampleFilter->SetOutputDirection(data_from_h5.ct_vol->GetDirection());
    resampleFilter->SetSize(data_from_h5.ct_vol->GetLargestPossibleRegion().GetSize());
    using NNInterpolatorType = itk::NearestNeighborInterpolateImageFunction<SegImageType, double>;
    NNInterpolatorType::Pointer nnInterpolator = NNInterpolatorType::New();
    resampleFilter->SetInterpolator(nnInterpolator);
    resampleFilter->Update();
    SegImageType::Pointer segResampled = resampleFilter->GetOutput();

    // Write out the resampled segmentation.
    {
        using WriterType = itk::ImageFileWriter<SegImageType>;
        WriterType::Pointer writer = WriterType::New();
        std::string resampledFilename = outDRRDir + "/resampled_segmentation.nii.gz";
        writer->SetFileName(resampledFilename);
        writer->SetInput(segResampled);
        try
        {
            writer->Update();
        }
        catch (itk::ExceptionObject &ex)
        {
            std::cerr << "Error writing resampled segmentation: " << ex << "\n";
        }
        std::cout << "Wrote resampled segmentation to " << resampledFilename << "\n";
    }

    // Create a binary mask: pixels equal to 1 become 1; others 0.
    using BinaryThresholdFilterType = itk::BinaryThresholdImageFilter<SegImageType, SegImageType>;
    BinaryThresholdFilterType::Pointer binaryFilter = BinaryThresholdFilterType::New();
    binaryFilter->SetInput(segResampled);
    binaryFilter->SetLowerThreshold(1);
    binaryFilter->SetUpperThreshold(1);
    binaryFilter->SetInsideValue(1);
    binaryFilter->SetOutsideValue(0);
    binaryFilter->Update();

    // Write out the binary mask.
    {
        using WriterType = itk::ImageFileWriter<SegImageType>;
        WriterType::Pointer writer = WriterType::New();
        std::string maskFilename = outDRRDir + "/binary_mask.nii.gz";
        writer->SetFileName(maskFilename);
        writer->SetInput(binaryFilter->GetOutput());
        try
        {
            writer->Update();
        }
        catch (itk::ExceptionObject &ex)
        {
            std::cerr << "Error writing binary mask: " << ex << "\n";
        }
        std::cout << "Wrote binary mask to " << maskFilename << "\n";
    }

    // Apply the binary mask to the CT volume.
    using MaskFilterType = itk::MaskImageFilter<CTImageType, SegImageType, CTImageType>;
    MaskFilterType::Pointer maskFilter = MaskFilterType::New();
    maskFilter->SetInput(data_from_h5.ct_vol);
    maskFilter->SetMaskImage(binaryFilter->GetOutput());
    maskFilter->SetOutsideValue(-1000.0f);
    maskFilter->Update();
    CTImageType::Pointer maskedCT = maskFilter->GetOutput();

    // Write out the masked CT volume (in HU).
    {
        using WriterType = itk::ImageFileWriter<CTImageType>;
        WriterType::Pointer writer = WriterType::New();
        std::string maskedCTFilename = outDRRDir + "/masked_CT_HU.nii.gz";
        writer->SetFileName(maskedCTFilename);
        writer->SetInput(maskedCT);
        try
        {
            writer->Update();
        }
        catch (itk::ExceptionObject &ex)
        {
            std::cerr << "Error writing masked CT (HU): " << ex << "\n";
        }
        std::cout << "Wrote masked CT (HU) to " << maskedCTFilename << "\n";
    }

    // Convert the masked CT volume from HU to linear attenuation.
    CTImageType::Pointer ct_lin_att = xreg::HUToLinAtt(maskedCT.GetPointer());

    // Write out the linear attenuation image.
    {
        using WriterType = itk::ImageFileWriter<CTImageType>;
        WriterType::Pointer writer = WriterType::New();
        std::string linAttFilename = outDRRDir + "/CT_lin_att.nii.gz";
        writer->SetFileName(linAttFilename);
        writer->SetInput(ct_lin_att);
        try
        {
            writer->Update();
        }
        catch (itk::ExceptionObject &ex)
        {
            std::cerr << "Error writing CT linear attenuation image: " << ex << "\n";
        }
        std::cout << "Wrote CT linear attenuation image to " << linAttFilename << "\n";
    }

    // Update the CT volume used for DRR generation.
    data_from_h5.ct_vol = ct_lin_att;
    // --- End Masking Step ---

    // --- Debug: Check statistics of the masked CT ---
    {
        using itk::ImageRegionConstIterator;
        ImageRegionConstIterator<CTImageType> it(data_from_h5.ct_vol, data_from_h5.ct_vol->GetLargestPossibleRegion());
        double minVal = std::numeric_limits<double>::max();
        double maxVal = -std::numeric_limits<double>::max();
        double sum = 0.0;
        size_t count = 0;
        for (it.GoToBegin(); !it.IsAtEnd(); ++it)
        {
            double val = it.Get();
            if (val < minVal)
                minVal = val;
            if (val > maxVal)
                maxVal = val;
            sum += val;
            ++count;
        }
        double meanVal = sum / count;
        std::cout << "Masked CT stats: min = " << minVal << ", max = " << maxVal << ", mean = " << meanVal << "\n";
    }

    // --- Uniform Sampling of Pose Offset ---
    std::mt19937 rng(std::random_device{}());
    const double lb_rot_deg = -5.0, ub_rot_deg = 5.0;
    const double lb_trans = -10.0, ub_trans = 10.0;
    const double deg2rad = M_PI / 180.0;
    double rx = std::uniform_real_distribution<double>(lb_rot_deg * deg2rad, ub_rot_deg * deg2rad)(rng);
    double ry = std::uniform_real_distribution<double>(lb_rot_deg * deg2rad, ub_rot_deg * deg2rad)(rng);
    double rz = std::uniform_real_distribution<double>(lb_rot_deg * deg2rad, ub_rot_deg * deg2rad)(rng);
    double tx = std::uniform_real_distribution<double>(lb_trans, ub_trans)(rng);
    double ty = std::uniform_real_distribution<double>(lb_trans, ub_trans)(rng);
    double tz = std::uniform_real_distribution<double>(lb_trans, ub_trans)(rng);

    FrameTransform pose_offset = EulerRotXYZTransXYZFrame(rx, ry, rz, tx, ty, tz);
    std::cout << "Uniformly sampled pose offset:\n"
              << pose_offset.matrix() << "\n";

    // For minimal test, use the offset as the new camera pose.
    FrameTransform new_cam_pose = pose_offset;
    std::cout << "New camera pose (for DRR generation):\n"
              << new_cam_pose.matrix() << "\n";

    // Update the projection data's camera transform.
    data_from_h5.pd.cam.extrins = new_cam_pose;

    // --- Run DRR Generation ---
    ProjPreProc proj_preproc;
    proj_preproc.input_projs = {data_from_h5.pd};

    // Disable log remapping.
    proj_preproc.params.no_log_remap = true;

    proj_preproc.set_debug_output_stream(std::cout, true);
    proj_preproc();

    itk::Image<float, 2>::Pointer drr_itk = proj_preproc.output_projs[0].img;
    cv::Mat drr = xreg::ShallowCopyItkToOpenCV(drr_itk.GetPointer());
    std::cout << "DRR image generated: " << drr.rows << " x " << drr.cols << "\n";

    // Debug: Print statistics of the DRR image.
    double drr_min, drr_max, drr_mean;
    {
        cv::minMaxLoc(drr, &drr_min, &drr_max);
        drr_mean = cv::mean(drr)[0];
        std::cout << "DRR stats: min = " << drr_min << ", max = " << drr_max << ", mean = " << drr_mean << "\n";
    }

    // Extract edges using OpenCV's Canny.
    cv::Mat drr_8u;
    drr.convertTo(drr_8u, CV_8U, 255.0);
    cv::Mat edges;
    cv::Canny(drr_8u, edges, 50, 150);

    // Save the DRR and edge images.
    const std::string drr_filename = outDRRDir + "/minimal_drr.png";
    const std::string edges_filename = outEdgesDir + "/minimal_edges.png";
    cv::imwrite(drr_filename, drr_8u);
    cv::imwrite(edges_filename, edges);

    std::cout << "Saved DRR image to " << drr_filename << "\n";
    std::cout << "Saved edge image to " << edges_filename << "\n";

    return 0;
}

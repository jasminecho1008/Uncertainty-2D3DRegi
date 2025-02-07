// minimal_data_gen.cpp
//
// This application reads an HDF5 file containing CT, segmentation, and projection data,
// samples a pose offset uniformly (without registration), applies that to the ground-truth
// camera pose, generates a DRR image using DeepDRR, extracts edges with Canny, and writes
// the results to disk.

#include <iostream>
#include <random>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

// xreg includes (adjust paths as needed)
#include "xregHDF5.h"
#include "xregH5ProjDataIO.h"
#include "xregITKIOUtils.h"
#include "xregProjPreProc.h"
#include "xregOpenCVUtils.h"
#include "xregRigidUtils.h"
#include "xregStringUtils.h"

// DeepDRR
#include "deepdrr/Projector.h"
#include "deepdrr/device/SimpleDevice.h"

// We assume that these types are defined by xreg:
using namespace xreg;

// A simplified version of the structure that holds the loaded data.
struct SamplingToolData
{
    itk::Image<float, 3>::Pointer ct_vol;
    itk::Image<unsigned char, 3>::Pointer seg_vol;
    ProjDataF32 pd;
    FrameTransform gt_cam_extrins_to_pelvis_vol;
};

// For simplicity we reuse the existing function that reads data from HDF5.
// (In your real code you likely already have ReadPelvisVolProjAndGtFromH5File.)
SamplingToolData ReadPelvisVolProjAndGtFromH5File(const std::string &h5_path,
                                                  const std::string &spec_id_str,
                                                  const size_t proj_idx,
                                                  const FrameTransform &trans_m,
                                                  std::ostream &vout)
{
    // For brevity, we assume this function is defined elsewhere and works as before.
    // (It reads the "proj-params" group, the "vol" group, etc.)
    SamplingToolData data;
    // ... (Use existing xreg code to populate 'data')
    // For this minimal example, we simply call the original function:
    data = ReadPelvisVolProjAndGtFromH5File(h5_path, spec_id_str, proj_idx, trans_m, vout);
    return data;
}

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
    const size_t num_samples = std::stoul(argv[4]);
    const std::string outDRRDir = argv[5];
    const std::string outEdgesDir = argv[6];

    // For simplicity, we use an identity transformation.
    FrameTransform trans_m = FrameTransform::Identity();

    // Read data from the HDF5 file.
    std::cout << "Reading HDF5 file: " << h5_file << std::endl;
    SamplingToolData data = ReadPelvisVolProjAndGtFromH5File(h5_file, patientID, proj_idx, trans_m, std::cout);

    // Uniformly sample one pose offset.
    // For example, sample rotations uniformly between -5 and 5 degrees,
    // and translations uniformly between -10 and 10 mm.
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

    // Build the offset transformation.
    FrameTransform pose_offset = EulerRotXYZTransXYZFrame(rx, ry, rz, tx, ty, tz);
    std::cout << "Uniformly sampled pose offset:\n"
              << pose_offset.matrix() << std::endl;

    // Apply the offset to the ground-truth camera transform.
    FrameTransform new_cam_pose = pose_offset * data.gt_cam_extrins_to_pelvis_vol;
    std::cout << "New camera pose:\n"
              << new_cam_pose.matrix() << std::endl;

    // Update the device view. (Assuming data.pd.device is a pointer to a SimpleDevice.)
    data.pd.device->set_view(new_cam_pose.point(), new_cam_pose.direction(), new_cam_pose.up(), 0.5);
    // (The exact API may vary; adjust to match your xreg/DeepDRR version.)

    // Create a projector for the CT volume and the new device.
    deepdrr::Projector projector({data.ct_vol /*, add other volumes if desired */}, data.pd.device);
    projector.initialize();

    // Generate a DRR image.
    cv::Mat drr = projector();
    if (drr.empty())
    {
        std::cerr << "DRR image generation failed." << std::endl;
        return 1;
    }
    std::cout << "DRR image generated with shape: " << drr.rows << " x " << drr.cols << std::endl;

    // Generate edges using a simple Canny detector.
    cv::Mat drr_8u;
    drr.convertTo(drr_8u, CV_8U, 255.0); // scale float image to 8-bit
    cv::Mat edges;
    cv::Canny(drr_8u, edges, 50, 150);

    // Save the DRR and edges images.
    const std::string drr_filename = outDRRDir + "/drr_sample.png";
    const std::string edges_filename = outEdgesDir + "/edges_sample.png";
    cv::imwrite(drr_filename, drr_8u);
    cv::imwrite(edges_filename, edges);

    std::cout << "DRR saved to " << drr_filename << "\n";
    std::cout << "Edges saved to " << edges_filename << "\n";

    return 0;
}

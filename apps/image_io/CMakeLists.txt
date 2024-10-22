# MIT License
#
# Copyright (c) 2020-2021 Robert Grupp
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

add_subdirectory(convert_resample_dicom)
add_subdirectory(report_dicom)
add_subdirectory(crop_vol)
add_subdirectory(remap_and_tile_proj_data)
add_subdirectory(extract_nii_from_proj_data)
add_subdirectory(add_lands_to_proj_data)
add_subdirectory(draw_xray_scene)
add_subdirectory(regi2d3d_replay)
add_subdirectory(convert_rad_raw_to_proj_data)
add_subdirectory(convert_sta_raw_to_itk)
add_subdirectory(convert_radiograph_dicom_to_proj_data)
add_subdirectory(make_video_from_image_dir)
add_subdirectory(remap_dicom_dir)
add_subdirectory(proj_cat)
add_subdirectory(proj_make_fiducial_world)
add_subdirectory(proj_est_orbital_rot)
add_subdirectory(convert_spare_projs_to_proj_data)


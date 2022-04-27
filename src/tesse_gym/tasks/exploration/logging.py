###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2020 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
# or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
# are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.
###################################################################################################

from typing import Tuple

import cv2
import numpy as np

from tesse_gym.core.logging import TESSEVideoWriter


class TESSEExplorationVideoWriter(TESSEVideoWriter):
    def step(self) -> None:
        """Create from after agent steps.

        The exploration exention adds the number of cells
        explored and the cell size as text on the video.
        """
        if self.write_frames:
            third_person, first_person, seg, depth = self.get_images(self.env)
            scale = third_person.shape[0] // first_person.shape[0]
            show_img = self.tile_imgs(
                self.resize_img(third_person, 2),
                self.resize_img(first_person, scale * 1 / 1.5),
                self.resize_img(seg, scale * 1 / 1.5),
                self.get_show_img(self.resize_img(depth, scale * 1 / 1.5)),
            )[..., (2, 1, 0)]

            if self.gym is not None and hasattr(self.gym, "visited_cells"):
                n_visited_cells = len(self.gym.visited_cells)
                cell_size = self.gym.cell_size
                relative_pose = self.gym.relative_pose
                show_img = show_img.copy()  # cv2.putText can't handle np array views
                self._add_text_to_img(
                    show_img,
                    f"Number of visited cells: {n_visited_cells}",
                    (10, 25),
                )
                self._add_text_to_img(
                    show_img, f"Cell size: {cell_size}m x {cell_size}m", (10, 55)
                )
                self._add_text_to_img(
                    show_img,
                    f"Relative Pose: ({relative_pose[0]:0.2f}, {relative_pose[1]:0.2f}, {relative_pose[2]:0.2f})",
                    (10, 85),
                )
                self.buffer.append(show_img)

    def _add_text_to_img(
        self, img: np.ndarray, text: str, position: Tuple[int, int]
    ) -> None:
        """Wrapper around cv2.putText."""
        cv2.putText(
            img,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,  # fontsize
            (0, 0, 255),  # red font
            2,  # line type
        )

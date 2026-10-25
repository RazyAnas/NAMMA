# How to film a room for NAMMA (from what broke on the 22 Oct video)

Every rule below comes from a failure we measured on the first video. Filming takes 1-2 minutes per room.

| Do this | Why (what went wrong on 22 Oct) |
|---|---|
| **Lock exposure and focus** before you start (iPhone: long-press the screen until "AE/AF LOCK"; Android: Pro mode, fix ISO/shutter) | The same white wall was brighter/darker on different passes, which made correct placements look wrong |
| **Turn slowly**: at least 3 seconds for a quarter turn | 4 of the 9 gaps between walk passes could not be followed: at a fast turn the frames had 0-2 points in common |
| **Never point at a blank wall up close.** When turning, keep a door frame, window, switch board or furniture in view | Plain white walls have nothing to match; that is where the camera solve lost track (frames 216, 488) |
| **Walk one continuous path, end where you started** | A closed loop lets us check and fix drift; our fix for the look-alike problem follows the video itself, so it must not jump |
| **Pass look-alike places slowly and include something unique** (a bin, a switch board, a sticker) in the same shot | Both corridor ends and the repeated grilled windows look alike; the solver mixed them up and put a window where a door is |
| **Hold the phone at chest height (~1.4 m), upright, steady** | Keeps the views comparable between passes; less blur |
| **Landscape, 4K 30 fps** (or 1x lens in landscape) | Portrait gave a narrow view: each photo covers less, so neighbouring photos overlap less and the viewer has more holes |
| **Walk about 1 m from the walls, camera looking across the room**, then one slow loop through the middle; pause 2 s at each corner and doorway | Coverage: anything never filmed becomes a hole that has to be generated (and flagged) |
| **Lights on, doors as they will be shown, nobody walking in the shot** | Moving people and changing doors break matching |
| **Several rooms**: film each room as its own loop, and film the doorway walk between rooms slowly in one take | Multi-room later: the doorway walk is what links two rooms |

Upload to the Kaggle dataset like the first video (`anasrazy/namma-room-2026-10-22` was the first; use a new dataset name per video).

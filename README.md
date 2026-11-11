# FixationFrames
Testing a few open questions after the FEAST paper

## Two Frames
If two frames are both moving at different directions or amplitudes, one larger than the other, what is the percept?

It depend on the frame that is closest to the dots? To prevent this simple explanation, the larger frame is made up of a texture of limited-life dots. And we also test each frame individually. One effect that was hinted at in one of the FEAST experiments is that if an edge frame (with nothing in the center) is larger, it's effect on the perception of the flashed probes decreases. To test this a little better, we add conditions with a single wire frame of various sizes, moving at 4 amplitudes.

This is all tested in the `frame_overlay.py` experiment, which has 2 blocks of 33 trials each.

Pilot show three things:

1. The smaller edge frame (4x4 dva) has a regular frame effect, the larger dot-texture frame (13x13 dva) has a smaller frame effect, but an effect nonetheless.
2. When both are combined, only the smaller edge frame seems to determine perception.
3. Increasing the frame size decreases the frame effect, perhaps less so for larger movement amplitudes (which bring one of the edges closer to the probes).

## Frame and motion extremes
If the size of the frame mediates the frame effect by virtue of larger edge frames having a gap between them and the probes, perhaps there are some other interesting phenomena as well.

Say we take a very narrow (0.5 dva) and very wide frame (14 dva), as well as a regular frame (7 dva). In this case they are monotone rectangles of 7 dva height, such that the distance to the edge is clear. We also use movement amplitudes of 4, 8, 12, 16 and 20 dva. 

For the 0.5 dva frame, increasing frame movement amplitudes increase the distance between frame and probes. For the two other frames, the edges would first get closer to the probes, then further again, although this turnaround would happen at different movement amplitudes.

Pilots seem to confirm that the distance between the edge of the frame and the probes dictates the frame effect.

This is captured in `frame_amplitude.py` (with expno==1).

## Gaze direction
However, while creating the previous paradigm, we noticed that some stimuli that should not have elicited any illusory offset between the probes, did so, as long as gaze was directed away from the probes, specifically, if if was closer to one of the edges of the frame.

So we added fixation control to the previous paradigm and for the 7 dva frame, we added some systematic offsets of gaze. The fixation dot was always above the stimulus at the same distance (a constant vertical offset), and for the previous conditions it was right above where the probes were flashed (so with no horizontal offset of fixation). For the new conditions, the horizontal offsets were 0, 2, 4, 6, 8, and 10 dva. To counter the possible guidance of the fixation dot itself as a stationary anchor for the illusion, we added two crosses that were not to be fixated but that were at the same vertical offset from the stimuli (both above and below, so they should be stronger anchors than just one fixation dot). These visual reference points were present in half the trials to test their effect.

With a movement amplitude of 20 dva the frame would be maximally visible at one extreme at a horizontal fixation offset of 10 dva (half the movement amplitude). Of course, the other end of the movement would then be less visible, so there could be some type of trade-off along these fixation offsets, so we did not included any larger movements.

Pilots show that the percept keeps increasing with increasing horizontal fixation offsets to the flashed probes. What we have not tested is what happens when fixation offsets become even larger. Two effects are likely competing: both the positional uncertainty of the frame and of the probes will increase. We suspect these would respectively decrease and increases the effect.

This is also captured in `frame_amplitude.py` (with expno==1).

## Positional uncertainty

One way to decrease the positional uncertainty of the frame is to decompose it into a cloud of dots that does not strictly move as one. There are various ways to do so, we combined three of these here. First, each dot could move on a path with a different angle. Second, each dot could move at a different amplitude. Third, each dot could have pauses in it's movement that are somewhat offset relative to the others. Each of these variables is distributed uniformly between two extremes, where the central values should always result in an average 4 dva movement.

For an average of 4 dva movement amplitude where one extreme movement amplitude is 0 dva, the other extreme has to be 8 dva. For any movement amplitude spreads larger than 2 times the average movement amplitude, a portion of dots moves in the opposite direction (creating incoherent dot clouds perhaps).

We 5 amplitude spreads: 4, 8, 12, 16 and 20 dva. For 4 and 8 dva none of the dots move in the opposite direction, for 12, 16 and 20 dva spreads, 16,67%, 25% and 30% of the dots move in the opposite direction to the average movement of the dot cloud (or to the movement of the center of the dot cloud).

This is all captured in `frame_compound.py`.

## Basic positional uncertainty

The flashed probes have more positional uncertainty than a continuously present probe. A vertical offset may enhance this further. It is possible that the size of the vertical offset between the extremes of the path of a continuously present probe -- that moves up and down -- can scale the frame effect.

We capture this in `frame_probes.py`.

Pilots show that the frame effect is largest in flashed probes, and the vertical offset has no (clear) effect there. For continuously presented probes, the vertical offset moderates the effect, which seems to saturate at 1.6 dva vertical offset (but never reaches the same size as the flashed probes). This suggests that in flashed probes the positional uncertainty is already at maximum, regardless of vertical offset, while in continuously present probes, a stationary probe has minimal positional certainty, and positional uncertainty increases with increasing vertical movement. But there is a limit to this. The frame motion then creates a frame effect, with a magnitude that depends on the positional uncertainty of the probe. What stands out to me is that (vertical) movement induces positional uncertainty.

Nothing about this is particularly surprising, but it's interesting that it can be shown in 24 - 36 trials per participant.



# nn_data_plugin

This plugin saves Rocket League 1v1 data to help train a neural network. The uploaded [data is available here](https://www.dropbox.com/sh/ac9ihejgfkqud72/AAAbUGoSvjgQfW9wk_4x_DxYa?dl=0). This tool records features like controller state which are not available in common replay databases.

I have not tested this in a few years so there may be issues. If you would like to test it and leave an issue showing the error, I will fix it. Otherwise, I'll be able to test/fix it myself in a few weeks when I'm back in the office.

## Getting Started

These instructions will get this plugin up and running on your local machine.

### Prerequisites

You will need to download Bakkes Mod from [here](https://bakkesmod.com/download.php), unzip the folder, and run BakkesMod.exe. This will create a bakkesmod folder located at `{STEAM INSTALLATION FOLDER}\steamapps\common\rocketleague\Binaries\Win32\bakkesmod\`

### Installing

After running BakkesMod.exe, download nn_data_plugin.dll from this github repo and place it in the folder `...\bakkesmod\plugins\`

Open plugins.cfg located at `...\bakkesmod\cfg\plugins.cfg` in a text editor and add the following line at the bottom:

```
nn_data_plugin
```

## Using the Plugin

The data recording process is fully automated. Every time you play a 1v1 match in casual or ranked modes, a new data file is saved to `...\bakkesmod\plugins\nn_data\` on your local machine and uploaded to the [shared dropbox folder](https://www.dropbox.com/sh/ac9ihejgfkqud72/AAAbUGoSvjgQfW9wk_4x_DxYa?dl=0). The name of the file contains your steam ID and a number which increments with each new upload from your account.

### Training a Neural Network

The python script `keras_sample.py` in this repo shows an example of how this data can be used to train a neural network to play Rocket League.

### Data Format

The list below describes each column of the data. All position values have an origin at the ball spawn location at the center of the arena.

```
1. my_team_ID (0 for blue team, 1 for orange team)
2. my_steamID
3. my_mmr (may be useful to take the highest MMR's to train the neural net)
4. my_score (as in goals scored)
5. my_x (x position of your vehicle, positive to the right, if you're in your spawn position)
6. my_y (y position of your vehicle, positive behind you, if you're in your spawn position)
7. my_z (z position of your vehicle, positive up)
8. my_rotx (rotational position of your vehicle)
9. my_roty
10. my_rotz
11. my_vx (velocity of your vehicle)
12. my_vy
13. my_vz
14. my_avx (angular velocity of your vehicle)
15. my_avy
16. my_avz
17. my_supersonic (boolean to tell when you are supersonic)
18. my_throttle (-1 for full reverse, 1 for full forward)
19. my_steer (-1 for full left, 1 for full right)
20. my_pitch (-1 for nose down, 1 for nose up)
21. my_yaw (-1 for full left, 1 for full right)
22. my_roll (-1 for roll left, 1 for roll right)
23. my_jump (true if jump button is pressed)
24. my_activateboost (true if boost is activated)
25. my_handbrake (true if handbrake is activated)
26. my_jumped (true if player has jumped)
27. my_boostamount
28. opponent_steamid (a unique ID is still created for cross-platform players)
29. opponent_mmr
30. opponent_score
31. opponent_x (positive to the opponent's right, if opponent is in their spawn position)
32. opponent_y (positive behind the opponent, if opponent is in their spawn position)
33. opponent_z
34. opponent_rotx
35. opponent_roty
36. opponent_rotz
37. opponent_vx
38. opponent_vy
39. opponent_vz
40. opponent_avx
41. opponent_avy
42. opponent_avz
43. opponent_supersonic
44. opponent_throttle
45. opponent_steer
46. opponent_pitch
47. opponent_yaw
48. opponent_roll
49. opponent_jump
50. opponent_activateboost
51. opponent_handbrake
52. opponent_jumped
53. opponent_boostamount
54. ball_x (same coordinate frame as my_x, my_y, my_z)
55. ball_y
56. ball_z
57. ball_vx
58. ball_vy
59. ball_vz
60. ball_avx
61. ball_avy
62. ball_avz
63. my_ball_touches (an integer which increments every time you touch the ball)
64. opponent_ball_touches
65. game_countdown (game countdown which updates every second, from 300 to 0)
```

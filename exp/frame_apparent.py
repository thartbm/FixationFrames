import time, os, random, sys, json, copy
import numpy as np
import pandas as pd
from psychopy import visual, core, event, monitors, tools
from psychopy.hardware import keyboard

# altenative keyboard read-out?
from pyglet.window import key

# fix a bug?
#import ctypes
#xlib = ctypes.cdll.LoadLibrary("libX11.so")
#xlib.XInitThreads()


#We really do need to know the operating system to answer this one. If you’re using ubuntu then this is the issue:
#To get psychtoolbox working for keyboard use you need the following steps to raise the priority of the experiment process. The idea is that to give you permission to do this without using super-user permissions to run your study (which would be bad for security) you need to add yourself to a group (e.g create a psychopy group) and then give that group permission to raise the priority of our process without being su:

#sudo groupadd --force psychopy
#sudo usermod -a -G psychopy $USER

#then do sudo nano /etc/security/limits.d/99-psychopylimits.conf and copy/paste in the following text to that file:

#@psychopy   -  nice       -20
#@psychopy   -  rtprio     50
#@psychopy   -  memlock    unlimited



# 600 dots
# 0.16667 dot life time
# 0.015 dot size

# expno: 1, 2, 3: dot frames / dot fields tasks with various numbers of trials


def doDotTrial(cfg):

    trialtype = cfg['blocks'][cfg['currentblock']]['trialtypes'][cfg['currenttrial']]
    trialdict = cfg['conditions'][trialtype]

    if 'record_timing' in trialdict.keys():
        record_timing = trialdict['record_timing']
    else:
        record_timing = False

    opacities = np.array([1]*len(cfg['hw']['dotfield']['dotlifetimes']))

    # straight up copies from the PsychoJS version:
    period = trialdict['period']
    #frequency = 1/copy.deepcopy(trialdict['period'])
    distance = trialdict['amplitude']

    # if trialdict['stimtype'] in ['barframe']:
    #     cfg['hw']['white_frame'].height = trialdict['barheight']
    #     cfg['hw']['gray_frame'].height = 16

    if 'framelag' in trialdict.keys():
        framelag = trialdict['framelag']
    else:
        framelag = 0
        trialdict['framelag'] = 0

    # determine which dots will always get set to be invisible:
    if 'dotfraction' in trialdict.keys():
        dotfraction = trialdict['dotfraction']
    else:
        dotfraction = 1.0
        # trialdict['dotfraction'] = 1.0

    if dotfraction < 1.0:
        #hiddendots = np.arange(0,len(opacities),step=len(opacities)/(1-dotfraction))
        #hiddendots = np.nonzero(np.floor(np.arange(len(opacities)) % (1/dotfraction)) > 0)[0]
        hiddendots = np.zeros(len(opacities))
        hiddendots[:np.round(len(opacities) * dotfraction).astype(int)] = 1
        np.random.shuffle(hiddendots)
        hiddendots = np.nonzero(hiddendots == 0)[0]
    else:
        hiddendots = np.array([], dtype=int)

    # flexibly set dot-life time:
    if 'dotlife' in trialdict.keys():
        maxdotlife = trialdict['dotlife']
    else:
        maxdotlife = cfg['hw']['dotfield']['maxdotlife']
        # trialdict['dotlife'] = maxdotlife

    if 'label' in trialdict.keys():
        label = trialdict['label']
    else:
        label = ''

    cfg['hw']['text'].text = label
    cfg['hw']['text'].pos = [4,4]

    # # change frequency and distance for static periods at the extremes:
    # if (0.35 - period) > 0:
    #     # make sure there is a 350 ms inter-flash interval
    #     extra_frames = int( np.ceil( (0.35 - period) / (1/60) ) * 2 )
    # else:
    #     extra_frames = 9

    # extra_frames = 9 + int( max(0, (0.35 - period) / (1/60) ) )

    # p = period + (extra_frames/60)
    p = period + (3/30)
    d = (distance/period) * p

    #print('period: %0.3f, p: %0.3f'%(period,p))
    #print('distance: %0.3f, d: %0.3f'%(distance,d))
    #print('speed: %0.3f, v: %0.3f'%(distance/period,d/p))


    #p = 1/f
    #print('p: %0.5f'%p)
    #print('d: %0.5f'%d)

    # DO THE TRIAL HERE
    trial_start_time = time.time()


    previous_frame_time = 0
    # # # # # # # # # #
    # WHILE NO RESPONSE

    frame_times = []
    frame_pos_X = []
    blue_on     = []
    red_on      = []

    frameoffset = [-8, -8]
    cfg['hw']['bluedot'].pos = [frameoffset[0], frameoffset[1]+1]
    cfg['hw']['reddot'].pos  = [frameoffset[0], frameoffset[1]-1]

    # we show a blank screen for 1/3 - 2.3 of a second (uniform dist):
    blank = 1/3 + (random.random() * 1/3)

    if cfg['expno'] in [2,3]:
        blank = p

    # the frame motion gets multiplied by -1 or 1:
    xfactor = [-1,1][random.randint(0,1)]

    if cfg['expno'] in [2,3]:
        xfactor = 1

    # the mouse response has a random offset between -3 and 3 degrees
    mouse_offset = (random.random() - 0.5) * 6

    waiting_for_response = True

    # if 'frameoffset' in trialdict.keys():
    #     cfg['hw']['bluedot'].pos=[-trialdict['frameoffset']/2,cfg['dot_offset']-cfg['stim_offsets'][1]]
    #     cfg['hw']['reddot'].pos=[-trialdict['frameoffset']/2,-cfg['dot_offset']-cfg['stim_offsets'][1]]
    #     frameoffset = [trialdict['frameoffset']/2, -cfg['stim_offsets'][1]]
    # else:
    #     cfg['hw']['bluedot'].pos=[0-cfg['stim_offsets'][0],cfg['dot_offset']-cfg['stim_offsets'][1]]
    #     cfg['hw']['reddot'].pos=[0-cfg['stim_offsets'][0],-cfg['dot_offset']-cfg['stim_offsets'][1]]
    #     frameoffset = [-cfg['stim_offsets'][0], -cfg['stim_offsets'][1]]



    while waiting_for_response:

        # blank screen of random length between 1/3 and 2.3 seconds
        while (time.time() - trial_start_time) < blank:
            event.clearEvents(eventType='mouse')
            event.clearEvents(eventType='keyboard')
            cfg['hw']['win'].flip()

        if cfg['expno'] in [2,3]:
            if (time.time() - trial_start_time) > (p * 9):
                reaction_time = 0
                waiting_for_response = False

        # on every frame:
        this_frame_time = time.time() - trial_start_time
        frame_time_elapsed = this_frame_time - previous_frame_time
        #print(round(1/frame_time_elapsed))

        # shorter variable for equations:
        t = this_frame_time

        # sawtooth, scaled from -0.5 to 0.5
        offsetX = abs( ( ((t/2) % p) - (p/2) ) * (2/p) ) - 0.5
        offsetX = offsetX * d

        flash_red  = False
        flash_blue = False
        flash_frame = False

        # flash any dots?
        if ( ((t + (1/30) + (framelag/30)) % (2*p)) < (1.75/30)):
            flash_red = True
        if ( ((t + (1/30) + p + (framelag/30)) % (2*p)) < (1.75/30) ):
            flash_blue = True

        # flash frame for apparent motion frame:
        if ( ((t + (1/30)) % (p/1)) < (2/30)):
            flash_frame = True

        # correct frame position:
        if (abs(offsetX) >= (distance/2)):
            offsetX = np.sign(offsetX) * (distance/2)
        else:
            flash_frame = False

        # flip offset according to invert percepts:
        offsetX = offsetX * xfactor

        # for all the conditions with dots, handle the dots:
        # if trialdict['stimtype'] in ['dotmovingframe','dotmotionframe','dotbackground','dotwindowframe']:
        #
        #     cfg['hw']['dotfield']['dotlifetimes'] += frame_time_elapsed
        #     idx = np.nonzero(cfg['hw']['dotfield']['dotlifetimes'] > maxdotlife)[0]
        #     #print(idx)
        #     cfg['hw']['dotfield']['dotlifetimes'][idx] -= maxdotlife
        #     cfg['hw']['dotfield']['xys'][idx,0] = np.random.random(size=len(idx)) - 0.5
        #
        #     xys = copy.deepcopy(cfg['hw']['dotfield']['xys'])
        #     xys[:,0] = xys[:,0] * (60 + cfg['maxamplitude'] - cfg['hw']['dotfield']['dotsize'])
        #     xys[:,1] = xys[:,1] * (15 - cfg['hw']['dotfield']['dotsize'])
        #
        #     opacities[:] = 1
        #     if (trialdict['stimtype'] == 'dotmovingframe'):
        #         opacities[np.nonzero(abs(xys[:,0]) > (7.5 - (cfg['hw']['dotfield']['dotsize']/2)))[0]] = 0
        #     if (trialdict['stimtype'] in ['dotmovingframe','dotmotionframe','dotbackground']):
        #         xys[:,0] += offsetX
        #     if (trialdict['stimtype'] == 'dotmotionframe'):
        #         opacities[np.nonzero(abs(xys[:,0]) > (7.5 - (cfg['hw']['dotfield']['dotsize']/2)))[0]] = 0
        #     if (trialdict['stimtype'] == 'dotwindowframe'):
        #         opacities[np.nonzero( abs(xys[:,0]-offsetX) > (7.5 - (cfg['hw']['dotfield']['dotsize']/2)) )[0]] = 0
        #
        #     if dotfraction < 1.0:
        #         opacities[hiddendots] = 0
        #     xys[:,0] = xys[:,0] - cfg['stim_offsets'][0]
        #     xys[:,1] = xys[:,1] - cfg['stim_offsets'][1]
        #     cfg['hw']['dotfield']['dotsarray'].setXYs(xys)
        #     cfg['hw']['dotfield']['dotsarray'].opacities = opacities
        #     cfg['hw']['dotfield']['dotsarray'].draw()

        # show frame for the classic and bar frames:
        if trialdict['stimtype'] in ['classicframe']:                              # barframe previously
            frame_pos = [offsetX+frameoffset[0], frameoffset[1]]
            # cfg['hw']['white_frame'].pos = frame_pos
            # cfg['hw']['white_frame'].draw()
            # cfg['hw']['gray_frame'].pos = frame_pos
            # cfg['hw']['gray_frame'].draw()
            cfg['hw']['frame'].pos = frame_pos
            cfg['hw']['frame'].draw()

        # flash frame for apparent motion frame:
        if (trialdict['stimtype'] == 'apparentframe') and flash_frame:
            # frame_pos = [offsetX-cfg['stim_offsets'][0], -cfg['stim_offsets'][1]]
            frame_pos = [offsetX+frameoffset[0], frameoffset[1]]
            # cfg['hw']['white_frame'].pos = frame_pos
            # cfg['hw']['white_frame'].draw()
            # cfg['hw']['gray_frame'].pos = frame_pos
            # cfg['hw']['gray_frame'].draw()
            cfg['hw']['frame'].pos = frame_pos
            cfg['hw']['frame'].draw()

        # flash the dots, if necessary:
        if flash_red:
            cfg['hw']['reddot'].draw()
        if flash_blue:
            cfg['hw']['bluedot'].draw()


        # in DEGREES:
        mousepos = cfg['hw']['mouse'].getPos()
        percept = (mousepos[0] + mouse_offset) / 4

        # blue is on top:
        # cfg['hw']['bluedot_ref'].pos = [percept+(2.5*cfg['stim_offsets'][0]),cfg['stim_offsets'][1]+9.5]
        # cfg['hw']['reddot_ref'].pos = [-percept+(2.5*cfg['stim_offsets'][0]),cfg['stim_offsets'][1]+6.5]
        cfg['hw']['bluedot_ref'].pos = [ (-1*frameoffset[0])+percept, (-1*frameoffset[1])+1 ]
        cfg['hw']['reddot_ref'].pos = [  (-1*frameoffset[0])-percept, (-1*frameoffset[1])-1 ]
        if cfg['expno'] in [2,3]:
            cfg['hw']['text'].draw()
        else:
            cfg['hw']['bluedot_ref'].draw()
            cfg['hw']['reddot_ref'].draw()

        cfg['hw']['win'].flip()

        previous_frame_time = this_frame_time

        frame_times += [this_frame_time]
        frame_pos_X += [offsetX]
        blue_on     += [flash_blue]
        red_on      += [flash_red]

        # key responses:
        keys = event.getKeys(keyList=['space','escape'])
        if len(keys):
            if 'space' in keys:
                waiting_for_response = False
                reaction_time = this_frame_time - blank
            if 'escape' in keys:
                cleanExit(cfg)

        if record_timing and ((this_frame_time - blank) >= 3.0):
            waiting_for_response = False


    if record_timing:
        pd.DataFrame({'time':frame_times,
                      'frameX':frame_pos_X,
                      'blue_flashed':blue_on,
                      'red_flashed':red_on}).to_csv('timing_data/%0.3fd_%0.3fs.csv'%(distance, period), index=False)
    else:
        response                = trialdict
        response['xfactor']     = xfactor
        response['RT']          = reaction_time
        response['percept_abs'] = percept
        response['percept_rel'] = percept/3
        response['percept_scl'] = (percept/3)*cfg['dot_offset']*2
        response['trial_start'] = trial_start_time
        response['blank']       = blank


        cfg['responses'] += [response]

    # cfg['hw']['white_frame'].height=15
    # cfg['hw']['gray_frame'].height=14

    cfg['hw']['win'].flip()

    return(cfg)

# def doMouseTrial(cfg):

#     trialtype = cfg['blocks'][cfg['currentblock']]['trialtypes'][cfg['currenttrial']]
#     trialdict = cfg['conditions'][trialtype]

#     # to output stimulus positions frame-by-frame:
#     if 'record_timing' in trialdict.keys():
#         record_timing = trialdict['record_timing']
#     else:
#         record_timing = False

#     # dot fields needs to have an array with opacities:
#     opacities = np.array([1]*len(cfg['hw']['dotfield']['dotlifetimes']))

#     # motion duration of 1 sweep (left-to-right or right-to-left)
#     period = trialdict['period']

#     # distance should probably by the maximum of the previous experiments (12 degrees)
#     distance = trialdict['amplitude']

#     # we don't want any frame lags here:
#     framelag = 0
#     trialdict['framelag'] = 0

#     gain = trialdict['gain'] # 1 or -1: frames moves with or against the hand

#     # # "metronome" needs this:
#     # # change frequency and distance for static periods at the extremes:
#     # if (0.35 - period) > 0:
#     #     # make sure there is a 350 ms inter-flash interval
#     #     extra_frames = int( np.ceil( (0.35 - period) / (1/60) ) * 2 )
#     # else:
#     #     extra_frames = 9

#     extra_frames = 9 + int( max(0, (0.35 - period) / (1/60) ) )

#     p = period + (extra_frames/60)
#     d = (distance/period) * p

#     #ppc = cfg['resolution'][0]/30.4 # how many pixels are in a cm on the tablet?
#     #groove_width = ppc * 15
#     #groove_lat_edge = ppc * 6.5
#     #groove_ant_edge = ppc * -6.6 # stay below this line to be in the groove

#     # we will get this stuff in degrees, for easier/faster code later on:
#     groove_width    = tools.monitorunittools.cm2deg(cm=15,  monitor=cfg['hw']['mon'])
#     groove_lat_edge = tools.monitorunittools.cm2deg(cm=6.5, monitor=cfg['hw']['mon'])
#     groove_ant_edge = tools.monitorunittools.cm2deg(cm=6.6, monitor=cfg['hw']['mon']) * -1

#     print(groove_ant_edge)

#     #top width: 30.4 cm (12 inch)
#     #side height: 21.8 cm
#     #middle height: 22.8
#     #groove height: 1 cm
#     #groove width: 15 cm

#     # mousepos = cfg['hw']['mouse'].getPos()
#     # percept = (mousepos[0] + mouse_offset) / 4


#     #comfortable speed: 0.6 s per movement


#     #print('period: %0.3f, p: %0.3f'%(period,p))
#     #print('distance: %0.3f, d: %0.3f'%(distance,d))
#     #print('speed: %0.3f, v: %0.3f'%(distance/period,d/p))


#     #p = 1/f
#     #print('p: %0.5f'%p)
#     #print('d: %0.5f'%d)

#     # DO THE TRIAL HERE
#     trial_start_time = time.time()


#     previous_frame_time = 0
#     # # # # # # # # # #
#     # WHILE NO RESPONSE

#     frame_times = []
#     frame_pos_X = []
#     blue_on     = []
#     red_on      = []

#     # we show a blank screen for 1/3 - 2.3 of a second (uniform dist):
#     blank = 1/3 + (random.random() * 1/3)

#     # the frame motion gets multiplied by -1 or 1:
#     xfactor = [-1,1][random.randint(0,1)]

#     waiting_for_response = True

#     percept = 0

#     while waiting_for_response:

#         # blank screen of random length between 1/3 and 2.3 seconds
#         while ((time.time() - trial_start_time) < blank) or ():
#             event.clearEvents(eventType='mouse')
#             event.clearEvents(eventType='keyboard')
#             cfg['hw']['win'].flip()

#         #event.clearEvents(eventType='keyboard')

#         #pixpos = getPixPos(cfg)
#         mousepos = cfg['hw']['mouse'].getPos()

#         #print(pixpos)
#         while mousepos[1] > groove_ant_edge:
#             mousepos = cfg['hw']['mouse'].getPos() # this is in DEGREES
#             #cfg['hw']['plus'].pos = mousepos
#             #cfg['hw']['plus'].draw()

#             #print(groove_ant_edge)
#             #print(mousepos)

#             # stylus not in groove:
#             cfg['hw']['text'].text='move the stylus into the groove...'
#             cfg['hw']['text'].draw()
#             cfg['hw']['win'].flip()
#             #print('above groove: %0.3f'%(groove_ant_edge))
#             keys = event.getKeys(keyList=['escape'])
#             if len(keys):
#                 if 'escape' in keys:
#                     cleanExit(cfg)


#             # RESET ANY VALUES?

#             #next()


#         # on every frame:
#         this_frame_time = time.time() - trial_start_time
#         frame_time_elapsed = this_frame_time - previous_frame_time
#         #print(round(1/frame_time_elapsed))

#         # shorter variable for equations:
#         t = this_frame_time



#         # sawtooth, scaled from -0.5 to 0.5
#         if trialdict['framecontrol'] == 'clock':
#             offsetX = abs( ( ((t/2) % p) - (p/2) ) * (2/p) ) - 0.5
#             offsetX = offsetX * d
#         if trialdict['framecontrol'] == 'mouse':
#             mousepos = cfg['hw']['mouse'].getPos()
#             #offsetX = min(abs(mousepos[0]), groove_lat_edge) * np.sign(mousepos[0]) * gain
#             offsetX = min(1, abs(mousepos[0])/groove_lat_edge) * np.sign(mousepos[0]) * gain * 0.5
#             offsetX = offsetX * distance
#             #print(abs(mousepos[0])/groove_lat_edge)


#         flash_red  = False
#         flash_blue = False
#         flash_frame = False

#         # flash any dots?
#         if ( ((t + (1/30)) % (2*p)) < (1.75/30)):
#             flash_red = True
#         if ( ((t + (1/30) + (p/1)) % (2*p)) < (1.75/30) ):
#             flash_blue = True

#         # instead of inverting the frame movement by xfactor,
#         # we invert the flash-order:
#         if xfactor == -1:
#             flash_red, flash_blue = flash_blue, flash_red

#         # flash frame for apparent motion frame:
#         #if ( ((t + (1/30)) % (p/1)) < (2/30)):
#         #    flash_frame = True

#         # correct frame position:
#         if trialdict['framecontrol'] == 'clock':
#             if (abs(offsetX) >= (distance/2)):
#                 offsetX = np.sign(offsetX) * (distance/2)
#             else:
#                 flash_frame = False

#         # flip offset according to invert percepts:
#         #offsetX = offsetX * xfactor
#         # this is now done by inverting flash order

#         # for all the conditions with dots, handle the dots:
#         if trialdict['stimtype'] in ['mousedotframe', 'mousedotbackground']:

#             cfg['hw']['dotfield']['dotlifetimes'] += frame_time_elapsed
#             idx = np.nonzero(cfg['hw']['dotfield']['dotlifetimes'] > cfg['hw']['dotfield']['maxdotlife'])[0]
#             #print(idx)
#             cfg['hw']['dotfield']['dotlifetimes'][idx] -= cfg['hw']['dotfield']['maxdotlife']
#             cfg['hw']['dotfield']['xys'][idx,0] = np.random.random(size=len(idx)) - 0.5

#             xys = copy.deepcopy(cfg['hw']['dotfield']['xys'])
#             xys[:,0] = xys[:,0] * (60 + cfg['maxamplitude'] - cfg['hw']['dotfield']['dotsize'])
#             xys[:,1] = xys[:,1] * (15 - cfg['hw']['dotfield']['dotsize'])

#             opacities[:] = 1
#             if (trialdict['stimtype'] == 'mousedotframe'):
#                 opacities[np.nonzero(abs(xys[:,0]) > (7.5 - (cfg['hw']['dotfield']['dotsize']/2)))[0]] = 0
#             xys[:,0] += offsetX
#             #if (trialdict['stimtype'] == 'dotmotionframe'):
#             #    opacities[np.nonzero(abs(xys[:,0]) > (7.5 - (cfg['hw']['dotfield']['dotsize']/2)))[0]] = 0
#             xys[:,0] = xys[:,0] - cfg['stim_offsets'][0]
#             xys[:,1] = xys[:,1] - cfg['stim_offsets'][1]
#             cfg['hw']['dotfield']['dotsarray'].setXYs(xys)
#             cfg['hw']['dotfield']['dotsarray'].opacities = opacities
#             cfg['hw']['dotfield']['dotsarray'].draw()

#         # show frame for the classic and bar frames:
#         if trialdict['stimtype'] in ['mouseclassicframe']:
#             frame_pos = [offsetX-cfg['stim_offsets'][0], -cfg['stim_offsets'][1]]
#             cfg['hw']['white_frame'].pos = frame_pos
#             cfg['hw']['white_frame'].draw()
#             cfg['hw']['gray_frame'].pos = frame_pos
#             cfg['hw']['gray_frame'].draw()


#         # flash the dots, if necessary:
#         if flash_red:
#             cfg['hw']['reddot'].draw()
#         if flash_blue:
#             cfg['hw']['bluedot'].draw()


#         # in DEGREES:
#         #mousepos = cfg['hw']['mouse'].getPos()
#         #percept = (mousepos[0] + mouse_offset) / 4

#         # blue is on top:
#         cfg['hw']['bluedot_ref'].pos = [percept+(2.5*cfg['stim_offsets'][0]),cfg['stim_offsets'][1]+9.5]
#         cfg['hw']['reddot_ref'].pos = [-percept+(2.5*cfg['stim_offsets'][0]),cfg['stim_offsets'][1]+6.5]
#         cfg['hw']['bluedot_ref'].draw()
#         cfg['hw']['reddot_ref'].draw()

#         cfg['hw']['win'].flip()

#         previous_frame_time = this_frame_time

#         frame_times += [this_frame_time]
#         frame_pos_X += [offsetX]
#         blue_on     += [flash_blue]
#         red_on      += [flash_red]

#         # key responses:
#         #keys = cfg['hw']['keyboard'].getKeys(keyList=['space','escape','a','left','end','d','right','pagedown'], waitRelease=False)
#         keys = event.getKeys(keyList=['space','escape'])
#         if len(keys):
#             if 'space' in keys:
#                 waiting_for_response = False
#                 reaction_time = this_frame_time - blank
#             if 'escape' in keys:
#                 # this causes an error,
#                 # so we wrap the whole experiment in try-except stuff
#                 cleanExit(cfg)

#         if any([cfg['hw']['keyboard'][key.LEFT], cfg['hw']['keyboard'][key.NUM_LEFT], cfg['hw']['keyboard'][key._1]]):
#             percept -= 0.025
#         if any([cfg['hw']['keyboard'][key.RIGHT], cfg['hw']['keyboard'][key.NUM_RIGHT], cfg['hw']['keyboard'][key._3]]):
#             percept += 0.025
#         if abs(percept) > 15: # twice the frame size should be a reasonable boundary
#             percept = np.sign(percept) * 15

#         if record_timing and ((this_frame_time - blank) >= 3.0):
#             waiting_for_response = False


#     if record_timing:
#         pd.DataFrame({'time':frame_times,
#                       'frameX':frame_pos_X,
#                       'blue_flashed':blue_on,
#                       'red_flashed':red_on}).to_csv('timing_data/%0.3fd_%0.3fs.csv'%(distance, period), index=False)
#     else:
#         response                = trialdict
#         response['xfactor']     = xfactor
#         response['RT']          = reaction_time
#         response['percept_abs'] = percept
#         response['percept_rel'] = percept/3
#         response['percept_scl'] = (percept/3)*cfg['dot_offset']*2

#         cfg['responses'] += [response]

#     cfg['hw']['white_frame'].height=15
#     cfg['hw']['gray_frame'].height=14

#     cfg['hw']['win'].flip()

#     return(cfg)


def runTasks(cfg):

    cfg = getMaxAmplitude(cfg)

    cfg['responses'] = []

    if not('currentblock' in cfg):
        cfg['currentblock'] = 0
    if not('currenttrial' in cfg):
        cfg['currenttrial'] = 0

    while cfg['currentblock'] < len(cfg['blocks']):

        # do the trials:
        cfg['currenttrial'] = 0

        showInstruction(cfg)

        while cfg['currenttrial'] < len(cfg['blocks'][cfg['currentblock']]['trialtypes']):

            trialtype = cfg['blocks'][cfg['currentblock']]['trialtypes'][cfg['currenttrial']]
            trialdict = cfg['conditions'][trialtype]


            if trialdict['stimtype'] in ['classicframe','apparentframe']:

                cfg = doDotTrial(cfg)
                saveCfg(cfg)

            else:

                print('unrecognized stimulus type: skipping trial')

            # if trialdict['stimtype'] in []:
            #
            #     cfg = doMouseTrial(cfg)
            #     saveCfg(cfg)

            cfg['currenttrial'] += 1

        cfg['currentblock'] += 1



    return(cfg)

def getStimuli(cfg, setup='tablet'):

    gammaGrid = np.array([[0., 1., 1., np.nan, np.nan, np.nan],
                          [0., 1., 1., np.nan, np.nan, np.nan],
                          [0., 1., 1., np.nan, np.nan, np.nan],
                          [0., 1., 1., np.nan, np.nan, np.nan]], dtype=float)
    # for vertical tablet setup:
    if setup == 'tablet':
        gammaGrid = np.array([[0., 136.42685, 1.7472667, np.nan, np.nan, np.nan],
                              [0.,  26.57937, 1.7472667, np.nan, np.nan, np.nan],
                              [0., 100.41914, 1.7472667, np.nan, np.nan, np.nan],
                              [0.,  9.118731, 1.7472667, np.nan, np.nan, np.nan]], dtype=float)
        waitBlanking = True
        resolution = [1680, 1050]
        size = [47, 29.6]
        distance = 60

    if setup == 'laptop':
    # for my laptop:
        waitBlanking = True
        resolution   = [1920, 1080]
        size = [34.5, 19.5]
        distance = 40


    mymonitor = monitors.Monitor(name='temp',
                                 distance=distance,
                                 width=size[0])
    mymonitor.setGammaGrid(gammaGrid)
    mymonitor.setSizePix(resolution)

    cfg['gammaGrid']    = list(gammaGrid.reshape([np.size(gammaGrid)]))
    cfg['waitBlanking'] = waitBlanking
    #cfg['resolution']   = resolution

    cfg['hw'] = {}

    # to be able to convert degrees back into pixels/cm
    cfg['hw']['mon'] = mymonitor

    # first set up the window and monitor:
    cfg['hw']['win'] = visual.Window( fullscr=True,
                                      size=resolution,
                                      units='deg',
                                      waitBlanking=waitBlanking,
                                      color=[0,0,0],
                                      monitor=mymonitor)

    res = cfg['hw']['win'].size
    cfg['resolution'] = [int(x) for x in list(res)]
    cfg['relResolution'] = [x / res[1] for x in res]

    cfg['stim_offsets'] = [4,2]

    #dot_offset = 6
    dot_offset = np.tan(np.pi/6)*6
    cfg['dot_offset'] = np.tan(np.pi/6)*6
    cfg['hw']['bluedot'] = visual.Circle(win=cfg['hw']['win'],
                                         units='deg',
                                         size=[1,1],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[-1,-1,1],
                                         pos=[0-cfg['stim_offsets'][0],dot_offset-cfg['stim_offsets'][1]])
    cfg['hw']['reddot'] = visual.Circle(win=cfg['hw']['win'],
                                         units='deg',
                                         size=[1,1],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[1,-1,-1],
                                         pos=[0-cfg['stim_offsets'][0],-dot_offset-cfg['stim_offsets'][1]])
    #np.tan(np.pi/6)*6

    ndots = 300
    maxdotlife = 1/5
    ypos = np.linspace(-0.5,0.5,ndots)
    random.shuffle(ypos)
    xys = [[random.random()-0.5,y] for y in ypos]
    #colors = [[-.25,-.25,-.25],[.25,.25,.25]] * 400
    colors = [[-.35,-.35,-.35],[.35,.35,.35]] * 150
    dotlifetimes = [random.random() * maxdotlife for x in range(ndots)]
    dotMask = np.ones([32,32])
    dotsize = 1

    dotsarray = visual.ElementArrayStim(win = cfg['hw']['win'],
                                        units='deg',
                                        fieldPos=(0,0),
                                        nElements=ndots,
                                        sizes=dotsize,
                                        colors=colors,
                                        xys=xys,
                                        elementMask=dotMask,
                                        elementTex=dotMask
                                        )

    dotfield = {}
    dotfield['maxdotlife']   = maxdotlife
    dotfield['dotlifetimes'] = np.array(dotlifetimes)
    dotfield['dotsarray']    = dotsarray
    dotfield['xys']          = np.array(xys)
    dotfield['dotsize']      = dotsize

    cfg['hw']['dotfield'] = dotfield

    cfg['hw']['white_frame'] = visual.Rect(win=cfg['hw']['win'],
                                           width=7,
                                           height=7,
                                           units='deg',
                                           lineColor=None,
                                           lineWidth=0,
                                           fillColor=[1,1,1])
    cfg['hw']['gray_frame'] =  visual.Rect(win=cfg['hw']['win'],
                                           width=6,
                                           height=6,
                                           units='deg',
                                           lineColor=None,
                                           lineWidth=0,
                                           fillColor=[0,0,0])

    cfg['hw']['frame'] = visual.ShapeStim(  win=cfg['hw']['win'],
                                            units='deg',
                                            colorSpace='rgb',
                                            lineColor=None,
                                            fillColor=[1,1,1],
                                            vertices=[[[-3.5,-3.5],[-3.5,3.5],[3.5,3.5],[3.5,-3.5],[-3.5,-3.5]],[[-3,-3],[-3,3],[3,3],[3,-3],[-3,-3]]]
                                            )                         


    cfg['hw']['bluedot_ref'] = visual.Circle(win=cfg['hw']['win'],
                                         units='deg',
                                         size=[1,1],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[-1,-1,1],
                                         pos=[0,0.20])
    cfg['hw']['reddot_ref'] = visual.Circle(win=cfg['hw']['win'],
                                         units='deg',
                                         size=[1,1],
                                         edges=180,
                                         lineWidth=0,
                                         fillColor=[1,-1,-1],
                                         pos=[0,-0.20])

    # we also want to set up a mouse object:
    cfg['hw']['mouse'] = event.Mouse(visible=False, newPos=None, win=cfg['hw']['win'])

    # keyboard is not an object, already accessible through psychopy.event
    ## WAIT... it is an object now!
    #print('done this...')
    #cfg['hw']['keyboard'] = keyboard.Keyboard()
    #print('but not this?')

    # pyglet keyboard system:
    cfg['hw']['keyboard'] = key.KeyStateHandler()
    cfg['hw']['win'].winHandle.push_handlers(cfg['hw']['keyboard'])

    # but it crashes the system...

    cfg['hw']['text'] = visual.TextStim(win=cfg['hw']['win'],
                                        text='Hello!'
                                        )
    cfg['hw']['plus'] = visual.TextStim(win=cfg['hw']['win'],
                                        text='+',
                                        units='deg'
                                        )

    return(cfg)


def getTasks(cfg):


    if cfg['expno']==1:

        condictionary = [

                         # classic frame with 10 lags
                         {'period':0.30, 'amplitude':4, 'stimtype':'classicframe', 'framelag': 0},
                         {'period':0.30, 'amplitude':4, 'stimtype':'classicframe', 'framelag': 2},
                         {'period':0.30, 'amplitude':4, 'stimtype':'classicframe', 'framelag': 3},
                         {'period':0.30, 'amplitude':4, 'stimtype':'classicframe', 'framelag': 4},
                         {'period':0.30, 'amplitude':4, 'stimtype':'classicframe', 'framelag': 5},
                         {'period':0.30, 'amplitude':4, 'stimtype':'classicframe', 'framelag': 6},
                         {'period':0.30, 'amplitude':4, 'stimtype':'classicframe', 'framelag': 7},
                         {'period':0.30, 'amplitude':4, 'stimtype':'classicframe', 'framelag': 8},
                         {'period':0.30, 'amplitude':4, 'stimtype':'classicframe', 'framelag': 9},
                         {'period':0.30, 'amplitude':4, 'stimtype':'classicframe', 'framelag': 10},
                         
                         # alternative amplitudes:
                         {'period':0.30, 'amplitude':3.2, 'stimtype':'classicframe', 'framelag': 0},
                         {'period':0.30, 'amplitude':2.4, 'stimtype':'classicframe', 'framelag': 0},
                         {'period':0.30, 'amplitude':1.6, 'stimtype':'classicframe', 'framelag': 0},
                         {'period':0.30, 'amplitude':0.8, 'stimtype':'classicframe', 'framelag': 0},

                         # apparent frame with the same 10 lags as before
                         {'period':0.30, 'amplitude':4, 'stimtype':'apparentframe', 'framelag': 0},
                         {'period':0.30, 'amplitude':4, 'stimtype':'apparentframe', 'framelag': 2},
                         {'period':0.30, 'amplitude':4, 'stimtype':'apparentframe', 'framelag': 3},
                         {'period':0.30, 'amplitude':4, 'stimtype':'apparentframe', 'framelag': 4},
                         {'period':0.30, 'amplitude':4, 'stimtype':'apparentframe', 'framelag': 5},
                         {'period':0.30, 'amplitude':4, 'stimtype':'apparentframe', 'framelag': 6},
                         {'period':0.30, 'amplitude':4, 'stimtype':'apparentframe', 'framelag': 7},
                         {'period':0.30, 'amplitude':4, 'stimtype':'apparentframe', 'framelag': 8},
                         {'period':0.30, 'amplitude':4, 'stimtype':'apparentframe', 'framelag': 9},
                         {'period':0.30, 'amplitude':4, 'stimtype':'apparentframe', 'framelag': 10},

                         # apparent frame with alternative amplitudes:
                         {'period':0.30, 'amplitude':3.2, 'stimtype':'apparentframe', 'framelag': 0},
                         {'period':0.30, 'amplitude':2.4, 'stimtype':'apparentframe', 'framelag': 0},
                         {'period':0.30, 'amplitude':1.6, 'stimtype':'apparentframe', 'framelag': 0},
                         {'period':0.30, 'amplitude':0.8, 'stimtype':'apparentframe', 'framelag': 0},


                         ]
                         # "frames" are virtually defined at 30 Hz (the monitor is 139 Hz or so)
                         # lag range: 0 to 300 ms
                         # frame movement of 300 ms, and a stationary period of 3 frames (3 x 33.3 ms) added for a half cycle of 300 ms

        return( dictToBlockTrials(cfg=cfg, condictionary=condictionary, nblocks=5, nrepetitions=2, shuffle=False) )
        # return( dictToBlockTrials(cfg=cfg, condictionary=condictionary, nblocks=3, nrepetitions=2) )

    



def run_exp(expno=1, setup='tablet', ID=np.nan):

    print(expno)

    cfg = {}
    cfg['expno'] = expno
    cfg['expstart'] = time.time()

    print(cfg)

    # get participant ID, set up data folder for them:
    # (function defined here)
    cfg = getParticipant(cfg, ID=ID)

    # define monitor Window for the current setup:
    # (function defined here)
    cfg = setWindow(cfg, setup=setup)

    # set up visual objects for the task:
    # (function defined per experiment)
    cfg = getStimuli(cfg)

    # set up blocks and trials/tasks within them:
    # (function defined per experiment)
    cfg = getTasks(cfg)
    # (function defined here)
    cfg = getMaxAmplitude(cfg)

    # try-catch statement in which we try to run all the tasks:
    # each trial saves its own data?
    # at the end a combined data file is produced?
    try:
        # run the tasks
        # (function defined here)
        cfg = runTasks(cfg)
    except Exception as e:
        # do this in case of error:
        print('there was an error:')
        print(e)
    else:
        # if there is no error: export data as csv
        # (function defined here)
        cfg = exportData(cfg)
    finally:
        # always do this:

        # save cfg, except for hardware related stuff (window object and stimuli pointing to it)
        # (function defined here)
        saveCfg(cfg)

        # shut down the window object
        # (function defined here)
        cleanExit(cfg)

def getParticipant(cfg, ID=np.nan, check_path=True):

    print(cfg)

    if np.isnan(ID):
        # we need to get an integer number as participant ID:
        IDnotANumber = True
    else:
        IDnotANumber = False
        cfg['ID'] = ID
        IDno = int(ID)

    # and we will only be happy when this is the case:
    while (IDnotANumber):
        # we ask for input:
        ID = input('Enter participant number: ')
        # and try to see if we can convert it to an integer
        try:
            IDno = int(ID)
            if isinstance(ID, int):
                pass # everything is already good
            # and if that integer really reflects the input
            if isinstance(ID, str):
                if not(ID == '%d'%(IDno)):
                    continue
            # only then are we satisfied:
            IDnotANumber = False
            # and store this in the cfg
            cfg['ID'] = IDno
        except Exception as err:
            print(err)
            # if it all doesn't work, we ask for input again...
            pass

    # set up folder's for groups and participants to store the data
    if check_path:
        for thisPath in ['../data', '../data/apparent' '../data/apparent/exp_%d'%(cfg['expno']), '../data/apparent/exp_%d/p%03d'%(cfg['expno'],cfg['ID'])]:
            if os.path.exists(thisPath):
                if not(os.path.isdir(thisPath)):
                    os.makedirs
                    sys.exit('"%s" should be a folder'%(thisPath))
                else:
                    # if participant folder exists, don't overwrite existing data?
                    if (thisPath == '../data/apparent/exp_%d/p%03d'%(cfg['expno'],cfg['ID'])):
                        sys.exit('participant already exists (crash recovery not implemented)')
            else:
                os.mkdir(thisPath)

        cfg['datadir'] = '../data/apparent/exp_%d/p%03d/'%(cfg['expno'],cfg['ID'])

    # we need to seed the random number generator:
    random.seed(99999 * IDno)

    return cfg

def setWindow(cfg, setup='tablet'):

    gammaGrid = np.array([[0., 1., 1., np.nan, np.nan, np.nan],
                          [0., 1., 1., np.nan, np.nan, np.nan],
                          [0., 1., 1., np.nan, np.nan, np.nan],
                          [0., 1., 1., np.nan, np.nan, np.nan]], dtype=float)
    # for vertical tablet setup:
    if setup == 'tablet':
        #gammaGrid = np.array([[0., 136.42685, 1.7472667, np.nan, np.nan, np.nan],
        #                      [0.,  26.57937, 1.7472667, np.nan, np.nan, np.nan],
        #                      [0., 100.41914, 1.7472667, np.nan, np.nan, np.nan],
        #                      [0.,  9.118731, 1.7472667, np.nan, np.nan, np.nan]], dtype=float)

        gammaGrid = np.array([[  0., 107.28029,  2.8466334, np.nan, np.nan, np.nan],
                              [  0.,  22.207165, 2.8466334, np.nan, np.nan, np.nan],
                              [  0.,  76.29962,  2.8466334, np.nan, np.nan, np.nan],
                              [  0.,   8.474467, 2.8466334, np.nan, np.nan, np.nan]], dtype=float)

        waitBlanking = True
        resolution = [1680, 1050]
        size = [47, 29.6]
        distance = 60

        wacomOneCM = resolution[0] / 31.1

    if setup == 'laptop':
    # for my laptop:
        waitBlanking = True
        resolution   = [1920, 1080]
        size = [34.5, 19.5]
        distance = 40

        wacomOneCM = resolution[0] / 29.5


    mymonitor = monitors.Monitor(name='temp',
                                 distance=distance,
                                 width=size[0])
    mymonitor.setGammaGrid(gammaGrid)
    mymonitor.setSizePix(resolution)

    cfg['gammaGrid']    = list(gammaGrid.reshape([np.size(gammaGrid)]))
    cfg['waitBlanking'] = waitBlanking
    #cfg['resolution']   = resolution

    cfg['hw'] = {}

    # to be able to convert degrees back into pixels/cm
    cfg['hw']['mon'] = mymonitor

    #cfg['hw']['groove'] = [ tools.monitorunittools.pix2deg( (resolution[0]/2) - (5*wacomOneCM), cfg['hw']['mon'], correctFlat=False),
    #                        tools.monitorunittools.pix2deg( (resolution[0]/2) + (5*wacomOneCM), cfg['hw']['mon'], correctFlat=False) ]

    cfg['trackextent'] = tools.monitorunittools.pix2deg( (5*wacomOneCM), cfg['hw']['mon'], correctFlat=False)

    # first set up the window and monitor:
    cfg['hw']['win'] = visual.Window( fullscr=True,
                                      size=resolution,
                                      units='deg',
                                      waitBlanking=waitBlanking,
                                      color=[0,0,0],
                                      monitor=mymonitor)
                                      # for anaglyphs: blendmode='add' !!!

    res = cfg['hw']['win'].size
    cfg['resolution'] = [int(x) for x in list(res)]
    cfg['relResolution'] = [x / res[1] for x in res]

    # print(cfg['resolution'])
    # print(cfg['relResolution'])

    return(cfg)

def getMaxAmplitude(cfg):

    maxamplitude = 0
    for cond in cfg['conditions']:
        maxamplitude = max(maxamplitude, cond['amplitude'])

    cfg['maxamplitude'] = maxamplitude

    return(cfg)

def showInstruction(cfg):

    cfg['hw']['text'].text = cfg['blocks'][cfg['currentblock']]['instruction']

    waiting_for_response = True

    while waiting_for_response:

        cfg['hw']['text'].draw()
        cfg['hw']['win'].flip()

        keys = event.getKeys(keyList=['enter', 'return', 'escape'])
        if len(keys):
            if 'enter' in keys:
                waiting_for_response = False
            if 'return' in keys:
                waiting_for_response = False
            if 'escape' in keys:
                cleanExit(cfg)

def cleanExit(cfg):

    cfg['expfinish'] = time.time()

    saveCfg(cfg)

    print('cfg stored as json')

    cfg['hw']['win'].close()

    return(cfg)

def dictToBlockTrials(cfg, condictionary, nblocks, nrepetitions, shuffle=True):

    cfg['conditions'] = condictionary

    blocks = []
    for block in range(nblocks):

        blockconditions = []

        for repeat in range(nrepetitions):
            trialtypes = list(range(len(condictionary)))
            if shuffle:
                random.shuffle(trialtypes)
            blockconditions += trialtypes

        blocks += [{'trialtypes':blockconditions,
                    'instruction':'get ready for block %d of %d\npress enter to start'%(block+1,nblocks)}]

    cfg['blocks'] = blocks

    return(cfg)

def saveCfg(cfg):

    scfg = copy.copy(cfg)
    del scfg['hw']

    with open('%scfg.json'%(cfg['datadir']), 'w') as fp:
        json.dump(scfg, fp,  indent=4)

def getPixPos(cfg):

    mousepos = cfg['hw']['mouse'].getPos() # this is in DEGREES
    pixpos = [tools.monitorunittools.deg2pix(mousepos[0], cfg['hw']['mon'], correctFlat=False),
              tools.monitorunittools.deg2pix(mousepos[1], cfg['hw']['mon'], correctFlat=False)]

    return(pixpos)

def exportData(cfg):

    responses = cfg['responses']

    # collect names of data:
    columnnames = []
    for response in responses:
        rks = list(response.keys())
        addthese = np.nonzero([not(rk in columnnames) for rk in rks])[0]
        # [x+1 if x >= 45 else x+5 for x in l]
        [columnnames.append(rks[idx]) for idx in range(len(addthese))]

    # make dict with columnnames as keys that are all empty lists:
    respdict = dict.fromkeys(columnnames)
    columnnames = list(respdict)
    for rk in respdict.keys():
        respdict[rk] = []

    #respdict = {}
    #for colname in columnnames:
    #    respdict[colname] = []

    # go through responses and collect all data into the dictionary:
    for response in responses:
        for colname in columnnames:
            if colname in list(response.keys()):
                respdict[colname] += [response[colname]]
            else:
                respdict[colname] += ['']

    #for rk in respdict.keys():
    #    print([rk, len(respdict[rk])])

    pd.DataFrame(respdict).to_csv('%sresponses.csv'%(cfg['datadir']), index=False)

    print('data exported to csv')

    return(cfg)

def foldout(a):
  # http://code.activestate.com/recipes/496807-list-of-all-combination-from-multiple-lists/

  r=[[]]
  for x in a:
    r = [ i + [y] for y in x for i in r ]

  return(r)
















print(sys.argv)

run_exp(expno=int(sys.argv[1]), setup='tablet', ID=int(sys.argv[2]))

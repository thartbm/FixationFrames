import time, os, random, sys, json, copy
import numpy as np
import pandas as pd
from psychopy import visual, core, event, monitors, tools
from psychopy.hardware import keyboard

# altenative keyboard read-out?
from pyglet.window import key


# calibration / eyetracking imports ----
sys.path.append(os.path.join('..', 'EyeTracking'))
from EyeTracking import EyeTracker
import math
import time
from glob import glob

from psychopy import gui, data
from psychopy.tools.coordinatetools import pol2cart, cart2pol
# ----




def doTrial(cfg):

    trialtype = cfg['blocks'][cfg['currentblock']]['trialtypes'][cfg['currenttrial']]
    # trialdict = cfg['conditions'][trialtype]
    trialdict = copy.deepcopy(cfg['conditions'][trialtype])


    if 'record_timing' in trialdict.keys():
        record_timing = trialdict['record_timing']
    else:
        record_timing = False

    # straight up copies from the PsychoJS version:
    period = trialdict['period']
    #frequency = 1/copy.deepcopy(trialdict['period'])
    # distance = trialdict['amplitude']


    if 'frameoffset' in trialdict.keys():
        frameoffset = trialdict['frameoffset']
    else:
        frameoffset = 0

    if 'label' in trialdict.keys():
        label = trialdict['label']
    else:
        label = ''
    # label = ''

    # cfg['hw']['text'].text = label

    # print('about to get wireframes properties')

    # WIREFRAME HANDLING

    if 'innersize' in trialdict.keys():
        innersize = trialdict['innersize']
    else:
        print('innersize must be specified for each trial')

    # print('got inner frame size')
    if innersize == None:
        drawinner = False
    else:
        cfg['hw']['innerframe'].size = [innersize,innersize]
        drawinner = True

    if 'inneramplitude' in trialdict.keys():
        inneramplitude = trialdict['inneramplitude']
    else:
        print('inneramplitude must be specified for each trial')

    if inneramplitude == None:
        inneramplitude = np.nan

    if 'outersize' in trialdict.keys():
        outersize = trialdict['outersize']
    else:
        print('outersize must be specified for each trial')

    # print('got outer frame size')
    if outersize == None:
        drawouter = False
    else:
        cfg['hw']['outerframe'].size = [outersize,outersize]
        drawouter = True
    
    if 'outeramplitude' in trialdict.keys():
        outeramplitude = trialdict['outeramplitude']
    else:
        print('outeramplitude must be specified for each trial')

    if outeramplitude == None:
        outeramplitude = np.nan



    # frame.size = [framewidth,7]

    # # change frequency and distance for static periods at the extremes:
    # if (0.35 - period) > 0:
    #     # make sure there is a 350 ms inter-flash interval
    #     extra_frames = int( np.ceil( (0.35 - period) / (1/30) ) )
    # else:
    #     extra_frames = 4

    extra_frames = 4 + int( max(0, (0.35 - period) / (1/30) ) )

    extra_time = (extra_frames/30)

    p = period + extra_time

    id = (inneramplitude/period) * p
    od = (outeramplitude/period) * p

    # print('id: '+str(id))
    # print('od: '+str(od))

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

    # we show a blank screen for 1/3 - 2.3 of a second (uniform dist):
    blank = 1/3 + (random.random() * 1/3)

    if cfg['expno'] == 2:
        blank = 1/3

    # the frame motion gets multiplied by -1 or 1:
    xfactor = [-1,1][random.randint(0,1)]

    if cfg['expno'] == 2:
        xfactor = 1

    # the mouse response has a random offset between -3 and 3 degrees
    mouse_offset = (random.random() - 0.5) * 6

    waiting_for_response = True

    stim_centre = [-8, -4]
    cfg['hw']['bluedot'].pos = [stim_centre[0], stim_centre[1]+1]
    cfg['hw']['reddot'].pos = [stim_centre[0], stim_centre[1]-1]
    frame_centre = stim_centre

    flashdot_centre = [8,8]

    cfg['hw']['fixdot'].pos = [frameoffset, 1]


    if cfg['expno'] == 2:
        cfg['hw']['text'].text = label
        cfg['hw']['text'].pos = [-10,8]

    # print('preparations done, entering while loop...')

    while waiting_for_response:

        if cfg['expno'] == 2:
            if (time.time() > (trial_start_time + 4.75)):
                reaction_time = 0
                waiting_for_response = False

        # blank screen of random length between 1/3 and 2.3 seconds
        while (time.time() - trial_start_time) < blank:
            event.clearEvents(eventType='mouse')
            event.clearEvents(eventType='keyboard')
            cfg['hw']['win'].flip()

        # on every frame:'frametype'

        this_frame_time = time.time() - trial_start_time
        frame_time_elapsed = this_frame_time - previous_frame_time
        #print(round(1/frame_time_elapsed))

        # shorter variable for equations:
        t = this_frame_time

        # sawtooth, scaled from -0.5 to 0.5
        offsetX = abs( ( ((t/2) % p) - (p/2) ) * (2/p) ) - 0.5

        # print(offsetX)

        innerOffsetX = offsetX * id
        outerOffsetX = offsetX * od

        # print(outerOffsetX)

        flash_red  = False
        flash_blue = False

        # flash any dots?
        if ( ((t + (1/30)) % (2*p)) < (1.75/30)):
            flash_red = True
        if ( ((t + (1/30) + (p/1)) % (2*p)) < (1.75/30) ):
            flash_blue = True


        # correct frame position:
        if (abs(innerOffsetX) >= (abs(inneramplitude)/2)):
            innerOffsetX = np.sign(innerOffsetX) * (abs(inneramplitude)/2)
        
        if (abs(outerOffsetX) >= (abs(outeramplitude)/2)):
            outerOffsetX = np.sign(outerOffsetX) * (abs(outeramplitude)/2)
        # print(outerOffsetX)

        # flip offset according to invert percepts:
        innerOffsetX = innerOffsetX * xfactor
        outerOffsetX = outerOffsetX * xfactor

        if cfg['expno'] in [2]:
            cfg['hw']['text'].draw()

        # cfg['hw']['fixdot'].draw()

        # # show frame for the classic frame:
        # frame_pos = [offsetX+stim_centre[0], stim_centre[1]]
        # frame.pos = frame_pos
        # frame.draw()

        cfg['hw']['innerframe'].pos = [innerOffsetX+stim_centre[0], stim_centre[1]]
        cfg['hw']['outerframe'].pos = [outerOffsetX+stim_centre[0], stim_centre[1]]
        if drawinner:
            cfg['hw']['innerframe'].draw()
        if drawouter:
            cfg['hw']['outerframe'].draw()

        if flash_red:
            cfg['hw']['reddot'].draw()
        if flash_blue:
            cfg['hw']['bluedot'].draw()


        # in DEGREES:
        mousepos = cfg['hw']['mouse'].getPos()
        percept = (mousepos[0] + mouse_offset) / 4

        # blue is on top:
        cfg['hw']['bluedot_ref'].pos = [percept+flashdot_centre[0], 1+flashdot_centre[1]]
        cfg['hw']['reddot_ref'].pos = [-percept+flashdot_centre[0],-1+flashdot_centre[1]]
        if cfg['expno'] == 2:
            pass
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
        pass
        # pd.DataFrame({'time':frame_times,
        #               'frameX':frame_pos_X,
        #               'blue_flashed':blue_on,
        #               'red_flashed':red_on}).to_csv('timing_data/%0.3fd_%0.3fs.csv'%(distance, period), index=False)
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

    #cfg['hw']['white_frame'].height=15
    #cfg['hw']['gray_frame'].height=14

    cfg['hw']['win'].flip()

    return(cfg)


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

            print([cfg['currenttrial'], len(cfg['blocks'][cfg['currentblock']]['trialtypes'])])

            trialtype = cfg['blocks'][cfg['currentblock']]['trialtypes'][cfg['currenttrial']]
            trialdict = cfg['conditions'][trialtype]

            if trialdict['stimtype'] in ['wireframes']:

                cfg = doTrial(cfg)
                saveCfg(cfg)

            cfg['currenttrial'] += 1

        cfg['currentblock'] += 1



    return(cfg)

def getStimuli(cfg, setup='tablet'):

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


    cfg['hw']['fixdot'] = visual.Circle(win=cfg['hw']['win'],
                                         units='deg',
                                         size=[.5,.5],
                                         edges=180,
                                         fillColor=[-1,-1,-1],
                                         lineWidth=5,
                                         lineColor=[1,1,1],
                                         pos=[0-cfg['stim_offsets'][0],-dot_offset-cfg['stim_offsets'][1]])

    cfg['hw']['frame'] = visual.Rect(win=cfg['hw']['win'],
                                           width=7,
                                           height=7,
                                           units='deg',
                                           lineColor=None,
                                           lineWidth=0,
                                           fillColor=[.4,.4,.4])


    cfg['hw']['innerframe'] = visual.Rect(win = cfg['hw']['win'],
                                          width = 4,
                                          height = 4,
                                          units = 'deg',
                                          lineColor = [1,1,1],
                                          lineWidth = 2,
                                          fillColor=None)

    cfg['hw']['outerframe'] = visual.Rect(win = cfg['hw']['win'],
                                          width = 9,
                                          height = 9,
                                          units = 'deg',
                                          lineColor = [1,1,1],
                                          lineWidth = 2,
                                          fillColor=None)
    

    cfg['hw']['white_bar'] = visual.Rect(win = cfg['hw']['win'],
                                         width = 1,
                                         height = 7,
                                         units = 'deg',
                                         lineColor = None,
                                         lineWidth = 0,
                                         fillColor=[1,1,1])


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

# def makeMultiFrame(cfg, width=7, height=7, pos=[0,0], col=[1,1,1], units='deg'):

#     xrange = list(np.linspace(0,width,parts*2))
#     vertices=[]
#     for part in range(parts):
#         lx = xrange.pop(0) - (width/2)
#         rx = xrange.pop(0) - (width/2)
#         uy = height/2
#         ly = height/-2
#         vertices += [[[lx,uy],[rx,uy],[rx,ly],[lx,ly],[lx,uy]]]

#     multiframe = visual.ShapeStim(  win=cfg['hw']['win'],
#                                     units=units,
#                                     pos=pos,
#                                     colorSpace='rgb',
#                                     lineColor=None,
#                                     fillColor=col,
#                                     vertices=vertices
#                                     )

#     return(multiframe)


# def makeBarFrame(cfg, innersize, outersize, pos=[0,0], col=[1,1,1], units='deg'):

    

#     vertices = [
#                   [
#                     [-(outersize[0]/2), (outersize[1]/2)],
#                     [ (outersize[0]/2), (outersize[1]/2)],
#                     [ (outersize[0]/2),-(outersize[1]/2)],
#                     [-(outersize[0]/2),-(outersize[1]/2)],
#                     [-(outersize[0]/2), (outersize[1]/2)]
#                   ],
#                   [
#                     [-(innersize[0]/2), (innersize[1]/2)],
#                     [ (innersize[0]/2), (innersize[1]/2)],
#                     [ (innersize[0]/2),-(innersize[1]/2)],
#                     [-(innersize[0]/2),-(innersize[1]/2)],
#                     [-(innersize[0]/2), (innersize[1]/2)]
#                   ]
#                ]

#     barframe = visual.ShapeStim(  win=cfg['hw']['win'],
#                                     units=units,
#                                     pos=pos,
#                                     colorSpace='rgb',
#                                     lineColor=None,
#                                     fillColor=col,
#                                     vertices=vertices
#                                     )

#     return(barframe)



def setupEyetracking(cfg):

    # use LiveTrack to check fixation on both eyes:
    # tracker = 'livetrack'
    trackEyes = [True,True]

    # do not store any files anywhere:
    filefolder = None
    filename = None

    colors = {'back' : [ 0, 0, 0],
              'both' : [-1,-1,-1]} # only for EyeLink
    
    ET = EyeTracker( tracker           = 'livetrack',
                     trackEyes         = trackEyes,
                     fixationWindow    = 2.0,
                     minFixDur         = 0.2,
                     fixTimeout        = 3.0,
                     psychopyWindow    = cfg['hw']['win'],
                     filefolder        = filefolder,
                     filename          = filename,
                     samplemode        = 'average',
                     calibrationpoints = 5,
                     colors            = colors ) # only for EyeLink

    cfg['hw']['tracker'] = ET

    return(cfg)

    



def getTasks(cfg):

    if cfg['expno']==1:

        # period: 1.0, 1/2, 1/3, 1/4, 1/5
        # amplit: 2.4, 4.8, 7.2, 9.6, 12
        # (speeds: 12, 24, 36, 48, 60 deg/s)

        # fixation in the middle
        # move whole frame leftward from fixation?
        condictionary = [


                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':4, 'outersize':13, 'outeramplitude':4, 'label':''},

                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':4, 'outersize':13, 'outeramplitude':3, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':4, 'outersize':13, 'outeramplitude':2, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':4, 'outersize':13, 'outeramplitude':1, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':4, 'outersize':13, 'outeramplitude':0, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':4, 'outersize':13, 'outeramplitude':-1, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':4, 'outersize':13, 'outeramplitude':-2, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':4, 'outersize':13, 'outeramplitude':-3, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':4, 'outersize':13, 'outeramplitude':-4, 'label':''},

                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':3, 'outersize':13, 'outeramplitude':4, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':2, 'outersize':13, 'outeramplitude':4, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':1, 'outersize':13, 'outeramplitude':4, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':0, 'outersize':13, 'outeramplitude':4, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':-1, 'outersize':13, 'outeramplitude':4, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':-2, 'outersize':13, 'outeramplitude':4, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':-3, 'outersize':13, 'outeramplitude':4, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':-4, 'outersize':13, 'outeramplitude':4, 'label':''},

                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':4, 'outersize':None, 'outeramplitude':None, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':3, 'outersize':None, 'outeramplitude':None, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':2, 'outersize':None, 'outeramplitude':None, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':4, 'inneramplitude':1, 'outersize':None, 'outeramplitude':None, 'label':''},

                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':None, 'inneramplitude':None, 'outersize':13, 'outeramplitude':4, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':None, 'inneramplitude':None, 'outersize':13, 'outeramplitude':3, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':None, 'inneramplitude':None, 'outersize':13, 'outeramplitude':2, 'label':''},
                         {'period':1/3, 'amplitude':1, 'stimtype':'wireframes', 'innersize':None, 'inneramplitude':None, 'outersize':13, 'outeramplitude':1, 'label':''},

                         ]

        return( dictToBlockTrials(cfg=cfg, condictionary=condictionary, nblocks=1, nrepetitions=3, shuffle=True) )
        # return( dictToBlockTrials(cfg=cfg, condictionary=condictionary, nblocks=1, nrepetitions=1, shuffle=False) )










def run_exp(expno=1, setup='tablet', ID=np.nan, eyetracking=False):

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

    # set up eye-tracking
    cfg['eyetracking'] = eyetracking
    if cfg['eyetracking']:
        cfg = setupEyetracking(cfg)

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

def getParticipant(cfg, ID=None, check_path=True):

    print(ID)

    if isinstance(ID, str):
        cfg['ID'] = ID

    while (ID == None):

        expInfo = {}
        expInfo['ID'] = ''

        #if askQuestions:
        dlg = gui.DlgFromDict(expInfo, title='Infos')
        if ID == None:
            if isinstance(expInfo['ID'], str):
                if len(expInfo['ID']) > 0:
                    ID = expInfo['ID']

    # set up folder's for groups and participants to store the data
    if check_path:
        # print('checking paths:')
        for thisPath in ['../data', '../data/wireframes', '../data/wireframes/exp_%d'%(cfg['expno']), '../data/wireframes/exp_%d/%s'%(cfg['expno'],cfg['ID'])]:
            # print(' - %s'%(thisPath))
            if os.path.exists(thisPath):
                if not(os.path.isdir(thisPath)):
                    # os.makedirs
                    sys.exit('"%s" should be a folder but is not'%(thisPath))
                else:
                    # if participant folder exists: do NOT overwrite existing data!
                    if (thisPath == '../data/wireframes/exp_%d/%s'%(cfg['expno'],cfg['ID'])):
                        # sys.exit('participant already exists (crash recovery not implemented)')
                        print('participant already exists (crash recovery not implemented)')
                        ID = None # trigger asking for ID again

            else:
                # print('making folder: "%s"', thisPath)
                os.mkdir(thisPath)

    

    # everything checks out, store in cfg:
    cfg['ID'] = ID

    # store data in folder for task / exp no / participant:
    cfg['datadir'] = '../data/wireframes/exp_%d/%s/'%(cfg['expno'],cfg['ID'])

    # we need to seed the random number generator:
    random.seed('wireframes' + ID)

    return cfg

def setWindow(cfg, setup='tablet'):

    gammaGrid = np.array([[0., 1., 1., np.nan, np.nan, np.nan],
                          [0., 1., 1., np.nan, np.nan, np.nan],
                          [0., 1., 1., np.nan, np.nan, np.nan],
                          [0., 1., 1., np.nan, np.nan, np.nan]], dtype=float)

    # # # # # # # # # # #
    # LIVETRACK specs

    # gammaGrid = np.array([ [  0., 135.44739,  2.4203537, np.nan, np.nan, np.nan  ],
    #                        [  0.,  27.722954, 2.4203537, np.nan, np.nan, np.nan  ],
    #                        [  0.,  97.999275, 2.4203537, np.nan, np.nan, np.nan  ],
    #                        [  0.,   9.235623, 2.4203537, np.nan, np.nan, np.nan  ]  ], dtype=np.float32)

    # resolution = [1920, 1080] # in pixels
    # size       = [59.8, 33.6] # in cm
    # distance   = 49.53 # in cm
    # screen     = 1  # index on the system: 0 = first monitor, 1 = second monitor, and so on

    # # # # # # # # # # #



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
        maxamplitude = max(maxamplitude, cond['amplitude']) # take the absolute?

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

    if cfg['eyetracking']:
        cfg['hw']['tracker'].shutdown()

    # garbage collection?

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

    # print(cfg['datadir'])

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

















if __name__ == "__main__":

    print(sys.argv)
    print(len(sys.argv))

    if len(sys.argv) > 1:
        expno = int(sys.argv[1])
    else:
        expno = 1
    
    if len(sys.argv) > 2:
        ID = sys.argv[2]
    else:
        ID = None # this will ask for an ID on the command line

    if len(sys.argv) > 3:
        eyetracking = sys.argv[3] # works if its already a bool
        if isinstance(eyetracking, str):
            eyetracking = eval(eyetracking)
    else:
        eyetracking = False

    run_exp( expno = expno, 
             setup='tablet',
             ID=ID, 
             eyetracking=eyetracking)
 # we need to get an integer number as participant ID:
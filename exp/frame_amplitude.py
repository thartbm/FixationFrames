import time, os, random, sys, json, copy
import numpy as np
import pandas as pd
from psychopy import visual, core, event, monitors, tools
from psychopy.hardware import keyboard

# altenative keyboard read-out?
from pyglet.window import key

#To get psychtoolbox working for keyboard use you need the following steps to raise the priority of the experiment process. The idea is that to give you permission to do this without using super-user permissions to run your study (which would be bad for security) you need to add yourself to a group (e.g create a psychopy group) and then give that group permission to raise the priority of our process without being su:

#sudo groupadd --force psychopy
#sudo usermod -a -G psychopy $USER

#then do sudo nano /etc/security/limits.d/99-psychopylimits.conf and copy/paste in the following text to that file:

#@psychopy   -  nice       -20
#@psychopy   -  rtprio     50
#@psychopy   -  memlock    unlimited






def doTrial(cfg):

    trialtype = cfg['blocks'][cfg['currentblock']]['trialtypes'][cfg['currenttrial']]
    trialdict = cfg['conditions'][trialtype]

    if 'record_timing' in trialdict.keys():
        record_timing = trialdict['record_timing']
    else:
        record_timing = False

    # straight up copies from the PsychoJS version:
    period = trialdict['period']
    #frequency = 1/copy.deepcopy(trialdict['period'])
    distance = trialdict['amplitude']


    if 'frameoffset' in trialdict.keys():
        frameoffset = trialdict['frameoffset']
    else:
        frameoffset = 0

    if 'label' in trialdict.keys():
        label = trialdict['label']
    else:
        label = ''


    # # change frequency and distance for static periods at the extremes:
    # if (0.35 - period) > 0:
    #     # make sure there is a 350 ms inter-flash interval
    #     extra_frames = int( np.ceil( (0.35 - period) / (1/30) ) )
    # else:
    #     extra_frames = 4

    extra_frames = 4 + int( max(0, (0.35 - period) / (1/30) ) )

    extra_time = (extra_frames/30)

    p = period + extra_time
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

    stim_centre = [-frameoffset, -8]
    cfg['hw']['bluedot'].pos = [stim_centre[0], stim_centre[1]+1]
    cfg['hw']['reddot'].pos = [stim_centre[0], stim_centre[1]-1]
    frame_centre = stim_centre

    flashdot_centre = [8,8]

    cfg['hw']['fixdot'].pos = [frameoffset, -2]


    if cfg['expno'] == 2:
        cfg['hw']['text'].text = label
        cfg['hw']['text'].pos = [4,4]

    # print('preparation done...')

    while waiting_for_response:

        if cfg['expno'] == 2:
            if (time.time() > (trial_start_time + 3.25)):
                reaction_time = 0
                waiting_for_response = False

        # blank screen of random length between 1/3 and 2.3 seconds
        while (time.time() - trial_start_time) < blank:
            event.clearEvents(eventType='mouse')
            event.clearEvents(eventType='keyboard')
            cfg['hw']['win'].flip()

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

        # flash any dots?
        if ( ((t + (1/30)) % (2*p)) < (1.75/30)):
            flash_red = True
        if ( ((t + (1/30) + (p/1)) % (2*p)) < (1.75/30) ):
            flash_blue = True

        # flash frame for apparent motion frame:
        if ( ((t + (1/30)) % (p/1)) < (2/30)):
            flash_frame = True

        # correct frame position:
        if (abs(offsetX) >= (distance/2)):
            offsetX = np.sign(offsetX) * (distance/2)

        # flip offset according to invert percepts:
        offsetX = offsetX * xfactor

        if cfg['expno'] == 2:
            cfg['hw']['text'].draw()

        # show frame for the classic frame:
        frame_pos = [offsetX+stim_centre[0], stim_centre[1]]
        cfg['hw']['frame'].pos = frame_pos
        cfg['hw']['frame'].draw()

        cfg['hw']['fixdot'].draw()

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

            trialtype = cfg['blocks'][cfg['currentblock']]['trialtypes'][cfg['currenttrial']]
            trialdict = cfg['conditions'][trialtype]

            if trialdict['stimtype'] in ['barframe','classicframe', 'timedframe']:

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
    # cfg['hw']['gray_frame'] =  visual.Rect(win=cfg['hw']['win'],
    #                                        width=6,
    #                                        height=6,
    #                                        units='deg',
    #                                        lineColor=None,
    #                                        lineWidth=0,
    #                                        fillColor=[0,0,0])

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


def getTasks(cfg):

    if cfg['expno']==1:

        # period: 1.0, 1/2, 1/3, 1/4, 1/5
        # amplit: 2.4, 4.8, 7.2, 9.6, 12
        # (speeds: 12, 24, 36, 48, 60 deg/s)

        # fixation in the middle
        # move whole frame leftward from fixation?
        condictionary = [

                         {'period':1/3, 'amplitude':4, 'stimtype':'classicframe', 'frameoffset':0},
                         {'period':1/3, 'amplitude':4, 'stimtype':'classicframe', 'frameoffset':8},

                         {'period':1/3, 'amplitude':12, 'stimtype':'classicframe', 'frameoffset':0},
                         {'period':1/3, 'amplitude':12, 'stimtype':'classicframe', 'frameoffset':8},

                         {'period':1/3, 'amplitude':20, 'stimtype':'classicframe', 'frameoffset':0},
                         {'period':1/3, 'amplitude':20, 'stimtype':'classicframe', 'frameoffset':8},


                         # ALTERNATIVE AMPLITUDES:
                         {'period':1/3, 'amplitude':3.2, 'stimtype':'classicframe', 'frameoffset':0},
                         {'period':1/3, 'amplitude':2.4, 'stimtype':'classicframe', 'frameoffset':0},
                         {'period':1/3, 'amplitude':1.6, 'stimtype':'classicframe', 'frameoffset':0},
                         {'period':1/3, 'amplitude':0.8, 'stimtype':'classicframe', 'frameoffset':0},

                         ]

        return( dictToBlockTrials(cfg=cfg, condictionary=condictionary, nblocks=1, nrepetitions=1, shuffle=False) )
        # return( dictToBlockTrials(cfg=cfg, condictionary=condictionary, nblocks=1, nrepetitions=1, shuffle=True) )









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
            # and store this in the cfggetPart
            cfg['ID'] = IDno
        except Exception as err:
            print(err)
            # if it all doesn't work, we ask for input again...
            pass

    # set up folder's for groups and participants to store the data
    if check_path:
        for thisPath in ['../data', '../data/amplitude', '../data/amplitude/exp_%d'%(cfg['expno']), '../data/amplitude/exp_%d/p%03d'%(cfg['expno'],cfg['ID'])]:
            if os.path.exists(thisPath):
                if not(os.path.isdir(thisPath)):
                    os.makedirs
                    sys.exit('"%s" should be a folder'%(thisPath))
                else:
                    # if participant folder exists, don't overwrite existing data?
                    if (thisPath == '../data/amplitude/exp_%d/p%03d'%(cfg['expno'],cfg['ID'])):
                        sys.exit('participant already exists (crash recovery not implemented)')
            else:
                os.mkdir(thisPath)

        cfg['datadir'] = '../data/amplitude/exp_%d/p%03d/'%(cfg['expno'],cfg['ID'])

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

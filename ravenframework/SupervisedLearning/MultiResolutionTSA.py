# Copyright 2017 Battelle Energy Alliance, LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
  Created on January 10, 2019

  @author: talbpaul, wangc
  Container to handle ROMs that are made of many sub-roms
"""
# standard libraries
import copy
import warnings
from collections import defaultdict, OrderedDict
import pprint

# external libraries
import abc
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
# internal libraries
from ..utils import utils, mathUtils, xmlUtils, randomUtils
from ..utils import InputData, InputTypes
from .SupervisedLearning import SupervisedLearning
from .SyntheticHistory import SyntheticHistory
# import pickle as pk # TODO remove me!
import os
#
#
#
#
class MultiResolutionTSA(SupervisedLearning):
  """ In addition to clusters for each history, interpolates between histories. """

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    spec = super().getInputSpecification()
    spec.description = r"""Provides an alternative way to build the ROM. In addition to clusters for each history, interpolates between histories."""
    specs = cls.addTSASpecs(specs)
    # segmenting and clustering
    segment = InputData.parameterInputFactory("Segment", strictMode=True,
                                              descr=r"""provides an alternative way to build the ROM. When
                                                this mode is enabled, the subspace of the ROM (e.g. ``time'') will be divided into segments as
                                                requested, then a distinct ROM will be trained on each of the segments. This is especially helpful if
                                                during the subspace the ROM representation of the signal changes significantly. For example, if the signal
                                                is different during summer and winter, then a signal can be divided and a distinct ROM trained on the
                                                segments. By default, no segmentation occurs.""")
    segmentGroups = InputTypes.makeEnumType('segmentGroup', 'segmentGroupType', ['decomposition'])
    segment.addParam('grouping', segmentGroups, descr=r"""enables the use of ROM subspace clustering in
        addition to segmenting if set to \xmlString{cluster}. If set to \xmlString{segment}, then performs
        segmentation without clustering. If clustering, then an additional node needs to be included in the
        \xmlNode{Segment} node.""", default='decomposition')
    spec.addSub(segment)
    return spec

  def __init__(self):
    """
      Constructor.
      @ In, kwargs, dict, initialization options
      @ Out, None
    """
    super().__init__()
    self.printTag = 'Interp. Cluster ROM'
    self._maxCycles = None # maximum number of cycles to run (default no limit)
    self._macroTemplate = SyntheticHistory()

  def setTemplateROM(self, romInfo):
    """
      Set the ROM that will be used in this grouping
      @ In, romInfo, dict, {'name':romName, 'modelInstance':romInstance}, the information used to set up template ROM
      @ Out, None
    """
    self._macroTemplate.setTemplateROM(romInfo)

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the model parameter input.
      @ In, paramInput, InputData.ParameterInput, the already parsed input.
      @ Out, None
    """
    super()._handleInput(paramInput)
    # notation: "pivotParameter" is for micro-steps (e.g. within-year, with a Clusters ROM representing each year)
    #           "macroParameter" is for macro-steps (e.g. from year to year)
    inputSpecs = paramInput.findFirst('Segment')
    try:
      self._macroParameter = inputSpecs.findFirst('macroParameter').value # pivot parameter for macro steps (e.g. years)
    except AttributeError:
      self.raiseAnError(IOError, '"interpolate" grouping requested but no <macroParameter> provided!')
    maxCycles = inputSpecs.findFirst('maxCycles')
    if maxCycles is not None:
      self._maxCycles = maxCycles.value
      self.raiseAMessage(f'Truncating macro parameter "{self._macroParameter}" to "{self._maxCycles}" successive steps.')
    self._macroSteps = {}                                               # collection of macro steps (e.g. each year)

    self._macroTemplate._handleInput(paramInput)            # example "yearly" SVL engine collection
  # passthrough to template
  def setAdditionalParams(self, params):
    """
      Sets additional parameters, usually when pickling or similar
      @ In, params, dict, params to set
      @ Out, setAdditionalParams, dict, additional params set
    """
    # max cycles
    for sub in params['paramInput'].subparts:
      if sub.name == 'maxCycles':
        self._maxCycles = sub.value
        self.raiseAMessage(f'Truncating macro parameter "{self._macroParameter}" to "{self._maxCycles}" successive step{"s" if self._maxCycles > 1 else ""}.')
        break
    for step, collection in self._macroSteps.items():
      # deepcopy is necessary because clusterEvalMode has to be popped out in collection
      collection.setAdditionalParams(copy.deepcopy(params))
    self._macroTemplate.setAdditionalParams(params)
    return super().setAdditionalParams(params)

  def setAssembledObjects(self, *args, **kwargs):
    """
      Sets up the assembled objects for this class.
      @ In, args, list, list of arguments
      @ In, kwargs, dict, dict of keyword arguments
      @ Out, None
    """
    self._macroTemplate.setAssembledObjects(*args, **kwargs)

  def readAssembledObjects(self):
    """
      Reads in assembled objects
      @ In, None
      @ Out, None
    """
    for step in self._macroSteps.values():
      step.readAssembledObjects()

  def writePointwiseData(self, writeTo):
    """
      Writes pointwise data about this ROM to the data object.
      @ In, writeTo, DataObject, data structure into which data should be written
      @ Out, None
    """
    pass # we don't have a good way to write any info right now
    # TODO this may be useful in the future
    # for year, model in self._macroSteps.items():
    #   print('')
    #   print('year indices',year)
    #   iS, iE, pS, pE = model._getSegmentData() # (i)ndex | (p)ivot, (S)tarts | (E)nds
    #   for i in range(len(iS)):
    #     print(i, iS[i], iE[i], pS[i], pE[i])

  def writeXML(self, writeTo, targets=None, skip=None):
    """
      Write out ARMA information
      @ In, writeTo, xmlUtils.StaticXmlElement, entity to write to
      @ In, targets, list, optional, unused
      @ In, skip, list, optional, unused
      @ Out, None
    """
    # write global information
    newNode = xmlUtils.StaticXmlElement('InterpolatedMultiyearROM')
    ## macro steps information
    newNode.getRoot().append(xmlUtils.newNode('MacroParameterID', text=self._macroParameter))
    newNode.getRoot().append(xmlUtils.newNode('MacroSteps', text=len(self._macroSteps)))
    newNode.getRoot().append(xmlUtils.newNode('MacroFirstStep', text=min(self._macroSteps)))
    newNode.getRoot().append(xmlUtils.newNode('MacroLastStep', text=max(self._macroSteps)))
    writeTo.getRoot().append(newNode.getRoot())
    # write info about EACH macro step
    main = writeTo.getRoot()
    for macroID, step in self._macroSteps.items():
      newNode = xmlUtils.StaticXmlElement('MacroStepROM', attrib={self._macroParameter: str(macroID)})
      step.writeXML(newNode, targets, skip)
      main.append(newNode.getRoot())

  ############### TRAINING ####################
  def train(self, tdict):
    """
      Trains the SVL and its supporting SVLs etc. Overwrites base class behavior due to
        special clustering and macro-step needs.
      @ In, trainDict, dict, dicitonary with training data
      @ Out, None
    """
    # tdict should have two parameters, the pivotParameter and the macroParameter -> one step per realization
    if self._macroParameter not in tdict:
      self.raiseAnError(IOError, 'The <macroParameter> "{}" was not found in the training DataObject! Training is not possible.'.format(self._macroParameter))
    ## TODO how to handle multiple realizations that aren't progressive, e.g. sites???
    # create each progressive step
    self._macroTemplate.readAssembledObjects()
    for macroID in tdict[self._macroParameter]:
      macroID = macroID[0]
      new = self._copyAssembledModel(self._macroTemplate)
      self._macroSteps[macroID] = new

    # train the existing steps
    for s, step in enumerate(self._macroSteps.values()):
      self.raiseADebug('Training Statepoint Year {} ...'.format(s))
      trainingData = dict((var, [tdict[var][s]]) for var in tdict.keys())
      step.train(trainingData, skipAssembly=True)
    self.raiseADebug('  Statepoints trained ')
    # interpolate missing steps
    self._interpolateSteps(tdict)
    self.amITrained = True

  def _interpolateSteps(self, trainingDict):
    """
      Master method for interpolating missing ROMs for steps
      @ In, trainingDict, dict, training information
      @ Out, None
    """
    # acquire interpolatable information
    exampleModel = list(self._macroSteps.values())[0] # example MACRO model (e.g. example year)
    ### TODO FIXME WORKING
    # the exampleModel has self._divisionInfo, but the macroTemplate does not!
    # HOWEVER, you can't currently retrain the macroTemplate, but the copied newModel
    # interpolated ROMS don't have the divisionInfo! Apparently there's a missing step in
    # here somewhere. Right now we raise an error with the already-trained Classifier,
    # maybe we can just reset that sucker.
    exampleRoms = exampleModel.getSegmentRoms(full=True)
    numSegments = len(exampleModel._clusterInfo['labels'])
    ## TODO can we reduce the number of unique transitions between clusters?
    ## For example, if the clusters look like this for years 1 and 3:
    ##   Year 1: A1 B1 A1 B1 A1 B1 A1 B1
    ##   Year 2: A2 B2 C2 A2 B2 C2 A2 B2
    ## the total transition combinations are: A1-A2, A1-B2, A1-C2, B1-A2, B1-B2, B1-C2
    ## which is 6 interpolations, but doing all of them we do 8 (or more in a full example).
    ## This could speed up the creation of the interpolated clustered ROMs possibly.
    ## Then, whenever we interpolate, we inquire from whom to whom?
    ## Wait, if you have more than 2 statepoints, this will probably not be worth it.
    ##         - rambling thoughts, talbpaul, 2019
    interps = [] # by segment, the interpreter to make new data
    ## NOTE interps[0] is the GLOBAL PARAMS interpolator!!!
    # statepoint years
    statepoints = list(self._macroSteps.keys())
    # all years
    allYears = list(range(min(statepoints), max(statepoints)))
    # what years are missing?
    missing = list(y for y in allYears if y not in statepoints)
    # if any years are missing, make the interpolators for each segment of each statepoint ROM
    if missing:
      # interpolate global features
      globalInterp = self._createSVLInterpolater(self._macroSteps, index='global')
      # interpolate each segment
      for segment in range(numSegments):
        interp = self._createSVLInterpolater(self._macroSteps, index=segment)
        # store interpolators, by segment
        interps.append(interp)
    self.raiseADebug('Interpolator trained')
    # interpolate new data
    ## now we have interpolators for every segment, so for each missing segment, we
    ## need to make a new Cluster model and assign its subsequence ROMs (pre-clustering).
    years = list(self._macroSteps.keys())
    models = []
    # TODO assuming integer years! And by years we mean MacroSteps, except we leave it as years right now!
    for y in range(min(years), max(years)):
      # don't replace statepoint years
      if y in years:
        self.raiseADebug('Year {} is a statepoint, so no interpolation needed.'.format(y))
        models.append(self._macroSteps[y])
        continue
      # otherwise, create new instances
      else:
        self.raiseADebug('Interpolating year {}'.format(y))
        newModel = self._interpolateSVL(trainingDict, exampleRoms, exampleModel, self._macroTemplate, numSegments, globalInterp, interps, y)
        models.append(newModel)
        self._macroSteps[y] = newModel

  def _createSVLInterpolater(self, modelDict, index=None):
    """
      Generates an interpolation object for a supervised engine
      @ In, modelDict, dict, models to interpolate
      @ In, index, int, optional, segment under consideration
      @ Out, interp, scipy.interp1d instance, interpolater
    """
    # index is the segment
    interp = {}
    df = None
    for step, model in modelDict.items():
      # step is the macro step, e.g. year
      if index is None:
        raise NotImplementedError
      # if the input model is not clustered (maybe impossible currently?), no segmenting consideration
      elif index == 'global':
        params = model._roms[0].parametrizeGlobalRomFeatures(model._romGlobalAdjustments)
      # otherwise, need to capture segment information as well as the global information
      else:
        params = model.getSegmentRoms(full=True)[index].getFundamentalFeatures(None)
      newDf = pd.DataFrame(params, index=[step])
      if df is None:
        df = newDf
      else:
        df = df._append(newDf)

    df.fillna(0.0) # FIXME is 0 really the best for all signals??
    # create interpolators
    interp['method'] = {}
    for header in params:
      interp['method'][header] = interp1d(df.index.values, df[header].values)
    # DEBUGG tools
    #fname = 'debug_statepoints_{}.pk'.format(index)
    #with open(fname, 'wb') as f:
    #  df.index.name = 'year'
    #  pk.dump(df, f)
    #print('DEBUGG interpolation data has been dumped to', fname)
    # END debugg
    return interp

  def _interpolateSVL(self, trainingDict, exampleRoms, exampleModel, template, N, globalInterp, segmentInterps, index):
    """
      interpolates a single engine for a single macro step (e.g. a single year)
      @ In, trainingDict, dict, dictionary with training data
      @ In, exampleRoms, list, segment roms from an interpolation setpoint year
      @ In, exampleModel, ROMCollection instance, master model from an interpolation setpoint year
      @ In, template, SupervisedLearning instance, template ROM for constructing new ones
      @ In, N, int, number of segments in play
      @ In, globalInterp, scipy.interp1d instance, interpolator for global settings
      @ In, segmentInterps, scipy.interp1d instance, interpolator for local settings
      @ In, index, int, year for which interpolation is being performed
      @ Out, newModel, SupervisedEngine instance, interpolated model
    """
    newModel = copy.deepcopy(exampleModel)
    segmentRoms = np.array([])
    for segment in range(N):
      params = dict((param, interp(index)) for param, interp in segmentInterps[segment]['method'].items())
      # DEBUGG, leave for future development
      #fname = 'debugg_interp_y{}_s{}.pk'.format(index, segment)
      #with open(fname, 'wb') as f:
      #  print('Dumping interpolated params to', fname)
      #  pk.dump(params, f)
      newRom = copy.deepcopy(exampleRoms[segment])
      inputs = newRom.readFundamentalFeatures(params)
      newRom.setFundamentalFeatures(inputs)
      segmentRoms = np.r_[segmentRoms, newRom]

    # add global params
    params = dict((param, interp(index)) for param, interp in globalInterp['method'].items())
    # DEBUGG, leave for future development
    #with open('debugg_interp_y{}_sglobal.pk'.format(index), 'wb') as f:
    #  pk.dump(params, f)

    # TODO assuming histories!
    pivotID = exampleModel._templateROM.pivotParameterID
    pivotValues = trainingDict[pivotID][0] # FIXME assumes pivot is the same for each year
    params = exampleModel._roms[0].setGlobalRomFeatures(params, pivotValues)
    newModel._romGlobalAdjustments = params
    # finish training by clustering
    newModel._clusterSegments(segmentRoms, exampleModel.divisions)
    newModel.amITrained = True
    return newModel

  def _copyAssembledModel(self, model):
    """
      Makes a copy of assembled model and re-performs assembling
      @ In, model, object, entity to copy
      @ Out, new, object, deepcopy of model
    """
    new = copy.deepcopy(model)
    # because assembled objects are excluded from deepcopy, add them back here
    new.setAssembledObjects({})
    return new

  ############### EVALUATING ####################
  def evaluate(self, edict):
    """
      Evaluate the set of interpolated models
      @ In, edict, dict, dictionary of evaluation parameters
      @ Out, result, dict, result of evaluation
    """
    # can we run SupervisedLearning.evaluate? Should this be an evaluateLocal?
    ## set up the results dict with the correct dimensionality
    ### actually, let's wait for the first sample to come in.
    self.raiseADebug('Evaluating interpolated ROM ...')
    results = None
    ## TODO set up right for ND??
    forcedMax = self._maxCycles if self._maxCycles is not None else np.inf
    numMacro = min(len(self._macroSteps), forcedMax)
    macroIndexValues = []
    for m, (macroStep, model) in enumerate(sorted(self._macroSteps.items(), key=lambda x: x[0])):
      if m + 1 > numMacro:
        break
      # m is an index of the macro step, in order of the macro values (e.g. in order of years)
      # macroStep is the actual macro step value (e.g. the year)
      # model is the ClusterROM instance for this macro step
      macroIndexValues.append(macroStep)
      self.raiseADebug(f' ... evaluating macro step "{macroStep}" ({m+1} / {numMacro})')
      subResult = model.evaluate(edict) # TODO same input for all macro steps? True for ARMA at least...
      indexMap = subResult.get('_indexMap', {})
      # if not set up yet, then frame results structure
      if results is None:
        results = {}
        finalIndexMap = indexMap # in case every rlz doesn't use same order, which would be lame
        pivotID = model._templateROM.pivotParameterID
        indices = set([pivotID, self._macroParameter])
        for indexes in finalIndexMap.values():
          indices.update(set(indexes))
        #pivotVals = subResult[pivotID]
        #numPivot = len(pivotVals)
        for target, values in subResult.items():
          # if an index, just set the values now # FIXME assuming always the same!
          ## FIXME thing is, they're not always the same, we're clustering, so sometimes there's diff num days!
          ## TODO for now, we simply require using a classifier that always has the same number of entries.
          if target in [pivotID, '_indexMap'] or target in indices:
            results[target] = values
          else:
            # TODO there's a strange behavior here where we have nested numpy arrays instead of
            # proper matrices sometimes; maybe it has to be this way for unequal clusters
            # As a result, we use the object dtype, onto which we can place a whole numpy array.
            results[target] = np.zeros([numMacro] + list(values.shape), dtype=object)
      # END setting up results structure, if needed
      # FIXME reshape in case indexMap is not the same as finalIndexMap?
      for target, values in subResult.items():
        if target in [pivotID, '_indexMap'] or target in indices:# indexMap:
          continue
        indexer = tuple([m] + [None]*len(values.shape))
        try:
          results[target][indexer] = values
        except ValueError:
          self.raiseAnError(RuntimeError, 'The shape of the histories along the pivot parameter is not consistent! Try using a clustering classifier that always returns the same number of clusters.')
    results['_indexMap'] = {} #finalIndexMap
    for target, vals in results.items():
      if target not in indices and target not in ['_indexMap']: # TODO get a list of meta vars?
        default = [] if vals.size == 1 else [pivotID]
        results['_indexMap'][target] = [self._macroParameter] + list(finalIndexMap.get(target, default))
    results[self._macroParameter] = macroIndexValues
    return results

  ############### DUMMY ####################
  # dummy methods that are required by SVL and not generally used
  def __confidenceLocal__(self, featureVals):
    """
      This should return an estimation of the quality of the prediction.
      This could be distance or probability or anything else, the type needs to be declared in the variable cls.qualityEstType
      @ In, featureVals, 2-D numpy array , [n_samples,n_features]
      @ Out, __confidenceLocal__, float, the confidence
    """
    pass

  def __resetLocal__(self):
    """
      Reset ROM. After this method the ROM should be described only by the initial parameter settings
      @ In, None
      @ Out, None
    """
    pass

  def __returnCurrentSettingLocal__(self):
    """
      Returns a dictionary with the parameters and their current values
      @ In, None
      @ Out, params, dict, dictionary of parameter names and current values
    """
    return {}

  def __returnInitialParametersLocal__(self):
    """
      Returns a dictionary with the parameters and their initial values
      @ In, None
      @ Out, params, dict,  dictionary of parameter names and initial values
    """
    return {}

  # Are private-ish so should not be called directly, so we don't implement them, as they don't fit the collection.
  def __evaluateLocal__(self, featureVals):
    """
      @ In,  featureVals, np.array, 2-D numpy array [n_samples,n_features]
      @ Out, targetVals , np.array, 1-D numpy array [n_samples]
    """
    pass

  def _train(self, featureVals, targetVals):
    """
      Perform training on samples in featureVals with responses y.
      For an one-class model, +1 or -1 is returned.
      @ In, featureVals, {array-like, sparse matrix}, shape=[n_samples, n_features],
        an array of input feature values
      @ Out, targetVals, array, shape = [n_samples], an array of output target
        associated with the corresponding points in featureVals
    """
    pass

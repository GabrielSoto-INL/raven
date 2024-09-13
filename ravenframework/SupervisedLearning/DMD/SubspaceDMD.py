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
  Created on July 21, 2024

  @author: Andrea Alfonsi
  Subspace DMD  model

"""
#Internal Modules (Lazy Importer)--------------------------------------------------------------------
from ...utils.importerUtils import importModuleLazy
#Internal Modules (Lazy Importer) End----------------------------------------------------------------

#External Modules------------------------------------------------------------------------------------
pydmd = importModuleLazy("pydmd")
#External Modules End--------------------------------------------------------------------------------

#Internal Modules------------------------------------------------------------------------------------
from ...SupervisedLearning.DMD import DMDBase
from ...utils import InputData, InputTypes
#Internal Modules End--------------------------------------------------------------------------------

class SubspaceDMD(DMDBase):
  """
    Subspace DMD (Parametric)
  """
  info = {'problemtype':'regression', 'normalize':False}

  def __init__(self):
    """
      Constructor that will appropriately initialize a supervised learning object
      @ In, None
      @ Out, None
    """
    super().__init__()

  @classmethod
  def getInputSpecification(cls):
    """
      Method to get a reference to a class that specifies the input data for
      class cls.
      @ In, cls, the class for which we are retrieving the specification
      @ Out, inputSpecification, InputData.ParameterInput, class to use for
        specifying input of cls.
    """
    specs = super(SubspaceDMD, cls).getInputSpecification()

    specs.description = r"""The \xmlString{SubspaceDMD} ROM (Subspace DMD) aimed to construct a time-dependent (or any other monotonic
    variable) surrogate model based on Subspace DMD
    This surrogate is aimed to perform a ``dimensionality reduction regression'', where, given time
    series (or any monotonic-dependent variable) of data, a set of modes each of which is associated
    with a fixed oscillation frequency and decay/growth rate is computed
    in order to represent the data-set.
    In order to use this Reduced Order Model, the \xmlNode{ROM} attribute
    \xmlAttr{subType} needs to be set equal to \xmlString{DMD}.
    \\
    Once the ROM  is trained (\textbf{Step} \xmlNode{RomTrainer}), its parameters/coefficients can be exported into an XML file
    via an \xmlNode{OutStream} of type \xmlAttr{Print}. The following variable/parameters  can be exported (i.e. \xmlNode{what} node
    in \xmlNode{OutStream} of type \xmlAttr{Print}):
    \begin{itemize}

      \item \xmlNode{svd\_rank}, see XML input specifications below
      \item \xmlNode{rescale\_mode}, see XML input specifications below
      \item \xmlNode{sorted\_eigs}, see XML input specifications below
      \item \xmlNode{features}, see XML input specifications below
      \item \xmlNode{timeScale}, XML node containing the array of the training time steps values
      \item \xmlNode{dmdTimeScale}, XML node containing the array of time scale in the DMD space (can be used as mapping
      between the  \xmlNode{timeScale} and \xmlNode{dmdTimeScale})
      \item \xmlNode{eigs}, XML node containing the eigenvalues (imaginary and real part)
      \item \xmlNode{amplitudes}, XML node containing the amplitudes (imaginary and real part)
      \item \xmlNode{modes}, XML node containing the dynamic modes (imaginary and real part)
    \end{itemize}"""
    specs.addSub(InputData.parameterInputFactory("svd_rank", contentType=InputTypes.IntegerType,
                                                 descr=r"""defines the truncation rank to be used for the SVD.
                                                 the rank for the truncation; if -1 all the columns of $U_q$ are used,
                                                 if svd\_rank is an integer grater than zero it is used as the number
                                                 of columns retained from U_q. $svd\_rank=0$ or float values are not supported
                                                 """, default=-1))
    specs.addSub(InputData.parameterInputFactory("rescale_mode", contentType=InputTypes.makeEnumType("rescale_mode", "RescaleModeType",
                                                                                                        ["auto", "None"]),
                                                 descr=r"""Scale Atilde as shown in 10.1016/j.jneumeth.2015.10.010 (section 2.4) before
                                                 computing its eigendecomposition. None means no rescaling, ‘auto’ means automatic
                                                 rescaling using singular values.
                                                 """, default="None"))
    specs.addSub(InputData.parameterInputFactory("sorted_eigs", contentType=InputTypes.makeEnumType("sorted_eigs", "SortedType",
                                                                                                        ["real", "abs", "False"]),
                                                 descr=r"""Sort eigenvalues (and modes/dynamics accordingly) by magnitude if sorted_eigs=``abs'',
                                                 by real part (and then by imaginary part to break ties) if sorted_eigs=``real''.
                                                 """, default="False"))
    return specs

  def _handleInput(self, paramInput):
    """
      Function to handle the common parts of the distribution parameter input.
      @ In, paramInput, ParameterInput, the already parsed input.
      @ Out, None
    """
    import pydmd
    from pydmd import SubspaceDMD
    super()._handleInput(paramInput)
    settings, notFound = paramInput.findNodesAndExtractValues(['svd_rank', 'rescale_mode','sorted_eigs'])
    # notFound must be empty
    assert(not notFound)
    # -1 no truncation, 0 optimal rank is computed, >1 truncation rank
    # if 0.0 < float < 1.0, computed rank is the number of the biggest sv needed to reach the energy identified by this float value
    self.dmdParams['svd_rank'] = settings.get('svd_rank')
    # how to rescale
    self.dmdParams['rescale_mode'] = settings.get('rescale_mode')
    if self.dmdParams['rescale_mode'] == 'None':
      self.dmdParams['rescale_mode'] = None
    # sorted eigs
    self.dmdParams['sorted_eigs'] = settings.get('sorted_eigs')
    if self.dmdParams['sorted_eigs'] == 'False':
      self.dmdParams['sorted_eigs'] = False

    self._dmdBase = SubspaceDMD
    # intialize the model
    self.initializeModel(self.dmdParams)



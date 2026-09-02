.. _reference:

*********
Reference
*********

Reference material is **information-oriented**: a dry, structured description of
the CHIMERApy API. It describes what each function does and the arguments it
takes, and nothing more. For worked examples see the :ref:`tutorials` and
:ref:`how-to-guides`; for background on the algorithm see :ref:`explanation`.

.. _reference-chimera:

The CHIMERA implementation
==========================

.. automodapi:: chimerapy.chimera

.. _reference-chimera-original:

The legacy implementation
=========================

A close port of the original IDL ``chimera.pro``, kept for cross-checking the
main implementation. It also performs the HMI magnetic-polarity classification.

.. automodapi:: chimerapy.chimera_original

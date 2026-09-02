# Delay boundary

The compact three-stage capacity probe passed the runtime budget with an
8-channel Delay candidate: 64,202 parameters, 291 KB of ONNX artifacts,
0.211 real-time factor, and 484.6 MB peak RSS for a 30-second CPU run.

The first direct wet-only inverse training attempt was stopped after nine
epochs. Although absolute ESR moved slightly, SI-SDR degraded to about -18 dB
on development clips. Long feedback echoes are not reliably identifiable from
wet audio alone; a residual spectral network therefore creates ringing instead
of recovering the original. No checkpoint was retained.

Delay stays out of the accepted wet-to-clean chain until one of these contracts
is explicit:

- a paired Clean/Wet analysis input is available for physical deconvolution;
- the user supplies the Delay parameters; or
- a separately sealed model proves improvement with a locked, auditable
  abstention policy.

This keeps the accepted Drive/Reverb model loss-preserving and avoids claiming
arbitrary long-delay inversion.

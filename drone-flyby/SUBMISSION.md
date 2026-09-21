# Publish the endpoint and run organizer validation

Official source: README_OFFICIAL.md, Validation and evaluation / Serve your endpoint.
Portal: https://cases.nordicaicup.com

## 1. Keep inference on the GPU workstation

Complete local tests in README.md first. Start the server and keep it running:

```bash
DRONE_CONFIG=configs/tracking.yaml \
DRONE_TRACE_DIR=results/remote_validation \
DRONE_TRACE_IMAGES=1 \
python api.py
```

Wait for startup to finish. In another terminal:

```bash
curl --fail http://127.0.0.1:9053/api
```

Do not train on the GPU while validating.

## 2. Expose through your cloud VM

If this API already runs directly on a publicly reachable GPU VM, open TCP 9053
for that VM and skip the SSH tunnel.

If inference runs on the university workstation, use your existing GCP VM as a
relay. On the VM, SSH remote forwarding must bind publicly. One configuration is:

```bash
# ON THE RELAY VM (requires sudo):
printf 'GatewayPorts clientspecified\nAllowTcpForwarding yes\n' | \
  sudo tee /etc/ssh/sshd_config.d/90-drone-forwarding.conf
sudo sshd -t
sudo systemctl reload ssh
```

These are VM administration commands; run them only on your intended relay VM.
If an existing SSH configuration already permits this, do not add conflicting
settings. An existing Match block can further restrict forwarding; inspect the
SSH service logs if the bind fails.

From the GPU workstation, using the actual VM login and public IP:

```bash
ssh -N -T \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R 0.0.0.0:9053:127.0.0.1:9053 \
  VM_USER@VM_PUBLIC_IP
```

Keep this terminal alive. -R is intentional: the public VM's port forwards to
your workstation's API. This does not put inference on the VM.

Allow TCP 9053 in the VM's GCP VPC firewall and any guest OS firewall. If your
existing allow-winter-soldier-api rule is the rule used for all three challenges,
retain its other ports when updating it, e.g.:

```bash
gcloud compute firewall-rules update allow-winter-soldier-api \
  --allow=tcp:9052,tcp:9053,tcp:9054
```

Check the active gcloud project and rule/VM target tags first; this example applies
only to that existing rule. It replaces that rule's allowed-port list. If it
contains other ports, retain those too. No firewall changes were made by this package.

Check on the VM:

```bash
ss -ltn | grep ':9053'
```

The listener needs 0.0.0.0:9053 (or the intended public-facing interface), not only
127.0.0.1:9053. Test from another machine/network:

```bash
curl --fail http://VM_PUBLIC_IP:9053/api
```

If bind fails, another server/tunnel may already own 9053. Identify it before
stopping anything; do not kill unrelated Python or SSH processes.

## 3. Prepare the exact endpoint

From drone-flyby/ on the workstation:

```bash
python scripts/preflight.py --skip-data \
  --public-url http://VM_PUBLIC_IP:9053/predict

python scripts/submission_info.py \
  --public-url http://VM_PUBLIC_IP:9053/predict \
  --output results/submission.json
```

Health reachability from your workstation does not prove organizer reachability.
The generated submission.json explicitly says prepared_not_submitted.

## 4. Verify, then Validate in the organizer portal

1. Open https://cases.nordicaicup.com and use the API key your team received.
   Keep that key out of code, shell history, traces and shared archives.
2. Select Drone Flyby and supply the FULL endpoint including /predict.
3. Run Verify and inspect its result. Its 30-second allowance is only a format
   check, not proof of the 3333-ms attempt timeout or 333-ms throughput target.
4. With no other attempt running, run Validation, then wait for the result.
   This is the 249-frame validation sequence, not the final Evaluation action.
5. Save the attempt ID, reported score, any reported class scores, and the exact
   local trace directory for that run. Keep its manifest.json with both weight
   hashes and resolved configuration.

Validation can be repeated according to the supplied README, one attempt at a
time. The separate final Evaluation uses a different 250-frame sequence and only
one completed attempt. Do not confuse those actions.

No organizer login/session/API key or trained model endpoint was available during
package preparation, and the portal could not be inspected through the available
web tool. Consequently there is no claimed remote submission, live UI walkthrough,
or remote score. The steps above follow the official supplied README; follow the
portal's actual form labels if they differ.

## 5. Inspect the recorded remote attempt

Stop the server cleanly only after the attempt finishes so its logger flushes.
Then use the actual trace path:

```bash
python scripts/review.py \
  --trace results/remote_validation/RUN_DIRECTORY/trace.jsonl \
  --expected-frames 249 \
  --output results/remote_review
```

Add --session if more than one session is present. Do not supply local Helsinki
annotations or run score_trace.py against Helsinki for this different remote
sequence. Ground-truth-free logs can diagnose latency, dropped frames, model
selection and registration, but cannot independently reproduce the hidden AP50.

# Report section: Model poisoning attack in FL and SFL

This experiment compared Federated Learning (FL) and Split Federated Learning (SFL Design 1) on MNIST using 10 clients over 50 communication rounds. The non-IID severity was varied using 1, 2, 4, and 6 labels per client. The optimiser was SGD with a base learning rate of 0.005. The global model was evaluated on the full MNIST test set after every aggregation round.

## Poisoning attack method
A model poisoning attack was introduced by selecting client 0 as the malicious client. The malicious client still performed normal local training, so it appeared to behave like an honest participant during the local training phase. However, before aggregation, its submitted model update was modified.

For an honest client, the local update is calculated as: `update = client_state - global_state`. For the malicious client, the submitted state was changed to: `global_state + (-5.0) × update`. Because the attack scale is negative, the malicious client sends an update in the opposite direction of honest learning. Because the magnitude is greater than one, the harmful update is amplified before FedAvg aggregation.

## Final-round attack impact
- 1 labels/client: FL baseline accuracy = 0.1144, FL attacked accuracy = 0.0980, FL accuracy drop = 0.0164; SFL baseline accuracy = 0.1144, SFL attacked accuracy = 0.0980, SFL accuracy drop = 0.0164.
- 2 labels/client: FL baseline accuracy = 0.4249, FL attacked accuracy = 0.0980, FL accuracy drop = 0.3269; SFL baseline accuracy = 0.4249, SFL attacked accuracy = 0.0980, SFL accuracy drop = 0.3269.
- 4 labels/client: FL baseline accuracy = 0.6400, FL attacked accuracy = 0.1024, FL accuracy drop = 0.5376; SFL baseline accuracy = 0.6400, SFL attacked accuracy = 0.1024, SFL accuracy drop = 0.5376.
- 6 labels/client: FL baseline accuracy = 0.5789, FL attacked accuracy = 0.1753, FL accuracy drop = 0.4036; SFL baseline accuracy = 0.5789, SFL attacked accuracy = 0.1753, SFL accuracy drop = 0.4036.

## Interpretation
The attack impact should be interpreted by comparing the baseline and attacked accuracy curves. If the attacked model has lower accuracy or higher loss than the baseline, the malicious update successfully degraded global learning. The one-label-per-client case may already perform close to random guessing, so the attack may not have much additional room to reduce accuracy. The 2, 4, and 6 labels/client cases are often more useful for measuring attack impact because the baseline model has more opportunity to learn before being damaged.
The global L2 distance between FL and reconstructed SFL remains an important equivalence check. If the L2 distance remains close to zero in both baseline and attack scenarios, it indicates that the SFL Design 1 implementation is behaving equivalently to FL under the same honest or poisoned update logic.
These results prepare the next stage of the project: defence. A defence method can now be evaluated by comparing three cases: no attack, attack without defence, and attack with defence. A successful defence should reduce the accuracy drop caused by the malicious client while preserving performance in the no-attack baseline.
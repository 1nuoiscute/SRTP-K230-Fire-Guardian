"""Checks for retention direction, exemption and teacher gradient isolation."""
import importlib.util
import unittest
if importlib.util.find_spec('torch') and importlib.util.find_spec('ultralytics'):
    import torch
    from fire_teacher_loss import retention_terms


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('ultralytics'), 'Optional Torch/Ultralytics unavailable')
class RetentionTests(unittest.TestCase):
    def pair(self):
        torch.manual_seed(42)
        teacher = dict(scores=torch.randn(2, 1, 9, requires_grad=True),
                       boxes=torch.randn(2, 16, 9, requires_grad=True))
        student = {k: v.detach().clone().requires_grad_(True) for k, v in teacher.items()}
        return student, teacher

    def test_matching_outputs_have_zero_loss(self):
        student, teacher = self.pair()
        terms = retention_terms(student, teacher, torch.tensor([True, True]), 4)
        self.assertTrue(all(abs(float(t.detach())) < 1e-6 for t in terms))

    def test_only_preserved_examples_receive_gradients(self):
        student, teacher = self.pair()
        student = {k: (v.detach() + .25 * torch.randn_like(v)).requires_grad_(True) for k, v in student.items()}
        terms = retention_terms(student, teacher, torch.tensor([True, False]), 4)
        self.assertTrue(all(float(t.detach()) > 0 for t in terms))
        sum(terms).backward()
        for key in student:
            self.assertGreater(float(student[key].grad[0].abs().sum()), 0)
            self.assertEqual(float(student[key].grad[1].abs().sum()), 0)
            self.assertIsNone(teacher[key].grad)

    def test_all_new_examples_are_exempt(self):
        student, teacher = self.pair()
        terms = retention_terms(student, teacher, torch.tensor([False, False]), 4)
        self.assertEqual(float(sum(terms).detach()), 0)
        sum(terms).backward()
        self.assertEqual(float(student['scores'].grad.abs().sum()), 0)
        self.assertIsNone(teacher['scores'].grad)


if __name__ == '__main__':
    unittest.main()

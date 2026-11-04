"""Public evidence algebra invariants that completion order can obscure in HTTP."""
import copy
import unittest
from cascade.temporal import TemporalAggregator


class TemporalTests(unittest.TestCase):
    def rows(self):
        return [dict(window_id=str(i),index=i,start_seconds=i,end_seconds=i+2.,
                     observed_end_seconds=i+2.,outcome='decided',decision='REAL',score=.1,
                     trace=[],source_frames=[]) for i in range(3)]

    def test_replay_completion_order_and_conflicting_terminal(self):
        rows=self.rows();rows[1].update(outcome='skipped',decision=None,score=None,reason='overload')
        rows[2].update(decision='FAKE',score=.8)
        summaries=[]
        for order in (rows,list(reversed(rows)),[rows[1],rows[0],rows[2]]):
            aggregator=TemporalAggregator()
            for row in order:
                aggregator.update(row);aggregator.update(copy.deepcopy(row))
            summaries.append(aggregator.snapshot(4.,finalization_reason='eof'))
            with self.assertRaisesRegex(ValueError,'Conflicting'):
                aggregator.update(dict(rows[2],score=.7))
        self.assertEqual(summaries[0],summaries[1]);self.assertEqual(summaries[1],summaries[2])
        a=summaries[0];self.assertEqual(a['state'],'FAKE_EVIDENCE')
        self.assertEqual(a['decided_seconds'],4.)
        self.assertEqual(a['gap_intervals'],[])
        self.assertEqual(a['unknown_intervals'],[[1,3.]])

    def test_real_only_scope_and_empty_failed_observations(self):
        a=TemporalAggregator()
        self.assertEqual(a.snapshot(0.)['state'],'INSUFFICIENT_EVIDENCE')
        a.update(self.rows()[0]);snapshot=a.snapshot(2.)
        self.assertEqual(snapshot['state'],'REAL_EVIDENCE_ONLY');self.assertTrue(snapshot['provisional'])
        self.assertEqual(a.snapshot(2.,finalization_reason='failure')['state'],'INSUFFICIENT_EVIDENCE')

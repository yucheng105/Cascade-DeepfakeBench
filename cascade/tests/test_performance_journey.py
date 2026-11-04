"""Phase 4 public HTTP timing and budget behavior, with actual Cascade routing."""
import time
import json
import http.client
import threading
from pathlib import Path
import unittest
from unittest.mock import patch

from cascade.tests import test_window_journey as fixtures


def responsive_child(connection, kind, args):
    from cascade.performance import rss
    connection.send(('ok',rss()))
    command,value=connection.recv()
    if command!='close':connection.send(('ok',value))
    connection.close()


class PerformanceJourneyTests(unittest.TestCase):
    journey = fixtures.WindowJourneyTests.journey
    short_events = staticmethod(fixtures.WindowJourneyTests.short_events)

    def test_scoped_timings_survive_http_download_and_delayed_persistence(self):
        from cascade.streaming import save_report
        class SlowWorker(fixtures.Worker):
            def infer(self, window):
                time.sleep(.02)
                return super().infer(window)
        def slow_save(output, report):
            time.sleep(.004)
            return save_report(output, report)
        with patch('cascade.streaming.save_report', side_effect=slow_save):
            with self.journey(self.short_events(), worker=SlowWorker) as (request, start, settled):
                job = settled(start())
                self.assertEqual(job['phase'], 'complete', job)
                _, report = request('GET', f"/jobs/{job['id']}/report.json")
                perf = report['performance']
                self.assertEqual(perf['schema_version'], 1)
                self.assertEqual(perf['slo_verdict'], 'NOT_EVALUATED')
                self.assertGreater(perf['persistence']['seconds_total'], .004)
                self.assertGreater(perf['persistence']['bytes_written_total'], 0)
                self.assertGreater(perf['source_read']['count'], 0)
                self.assertEqual(perf['latency_by_scope']['since_window_admission']['count'], 4)
                for row in report['windows']:
                    self.assertGreaterEqual(row['timing']['worker_roundtrip_seconds'], .02)
                    self.assertGreaterEqual(row['timing']['persistence_seconds'], .004)
                    self.assertGreaterEqual(row['timing']['queue_seconds'], 0.)
                    self.assertIsNone(row['timing']['tensor_seconds'])
                    self.assertIn('unavailable', row['timing'])
                self.assertEqual(job['report']['aggregate'], report['aggregate'])

    def test_metadata_limit_keeps_completed_evidence_and_restart_report(self):
        # Tiny configured budget forces the real report path to exhaust, without
        # simulating a storage exception. Reserve permits terminal failure metadata.
        from cascade.demo import make_server
        servers=[]
        def capture(*args,**kwargs):
            server=make_server(*args,**kwargs);servers.append(server);return server
        with patch('cascade.demo.make_server',side_effect=capture):
            with self.journey(self.short_events(), max_report_bytes=12000) as (request, start, settled):
                job=settled(start())
                self.assertEqual(job['phase'],'failed',job)
                self.assertIn('metadata budget',job['message'])
                _,report=request('GET',f"/jobs/{job['id']}/report.json")
                self.assertEqual(report['phase'],'failed')
                self.assertEqual(report['aggregate']['finalization_reason'],'failure')
                self.assertTrue(any(w['outcome']=='decided' for w in report['windows']))
                self.assertFalse(any(w['outcome'] in ('waiting','processing','collecting') for w in report['windows']))
                self.assertLess(len(json.dumps(report).encode()),12000+4*1024*1024)
                restarted=make_server(servers[0].demo_state.root,port=0,device='cpu')
                thread=threading.Thread(target=restarted.serve_forever,daemon=True);thread.start()
                try:
                    conn=http.client.HTTPConnection('127.0.0.1',restarted.server_port,timeout=5)
                    conn.request('GET','/api/jobs/'+job['id'])
                    loaded=json.loads(conn.getresponse().read());conn.close()
                    self.assertEqual(loaded['phase'],'failed')
                    self.assertEqual(loaded['report']['aggregate'],report['aggregate'])
                    # API snapshots intentionally strip geometry and normalize
                    # missing profiles. Compare complete evidence via download.
                    conn=http.client.HTTPConnection('127.0.0.1',restarted.server_port,timeout=5)
                    conn.request('GET',f"/jobs/{job['id']}/report.json")
                    restored=json.loads(conn.getresponse().read());conn.close()
                    self.assertEqual(restored['windows'],report['windows'])
                finally:
                    restarted.shutdown();restarted.server_close();thread.join(2)

    def test_oversized_worker_metadata_is_not_accepted_as_terminal(self):
        class HugeWorker(fixtures.Worker):
            def infer(self,window):
                return dict(super().infer(window),unexpected='x'*600000)
        with self.journey(self.short_events(),worker=HugeWorker) as (_,start,settled):
            job=settled(start())
            self.assertEqual(job['phase'],'failed')
            self.assertIn('Window metadata budget',job['message'])
            self.assertFalse(any(w['outcome']=='decided' for w in job['report']['windows']))

    def test_final_snapshot_budget_failure_does_not_lock_eof_finalization(self):
        from cascade.streaming import save_report
        from cascade.preprocessing import MetadataBudgetExceeded
        for failure in (MetadataBudgetExceeded('final metadata budget exceeded'), OSError('final CSV unavailable')):
            with self.subTest(failure=type(failure).__name__):
                def reject_complete(output,report):
                    if report['phase']=='complete':
                        # Model a final CSV failure after the JSON replacement.
                        if isinstance(failure,OSError):save_report(output,report)
                        raise failure
                    return save_report(output,report)
                with patch('cascade.streaming.save_report',side_effect=reject_complete):
                    with self.journey(self.short_events()) as (request,start,settled):
                        job=settled(start())
                        self.assertEqual(job['phase'],'failed')
                        _,report=request('GET',f"/jobs/{job['id']}/report.json")
                        self.assertEqual(report['phase'],'failed')
                        self.assertEqual(report['aggregate']['finalization_reason'],'failure')
                        self.assertEqual(report['summary']['decided'],4)

    def test_poll_cache_keeps_full_download_and_refreshes_changed_report(self):
        from cascade.demo import read_json, make_server
        servers=[]
        def capture(*args,**kwargs):
            server=make_server(*args,**kwargs);servers.append(server);return server
        reads=[]
        def observed_read(path):
            reads.append(Path(path))
            return read_json(path)
        with patch('cascade.demo.make_server',side_effect=capture):
            with self.journey(self.short_events()) as (request,start,settled):
                job=settled(start())
                identity=job['id']
                with patch('cascade.demo.read_json',side_effect=observed_read):
                    snapshots=[request('GET','/api/jobs/'+identity)[1]['report'] for _ in range(4)]
                self.assertLessEqual(len(reads),1)
                self.assertTrue(all(r==snapshots[0] for r in snapshots))
                _,full=request('GET',f'/jobs/{identity}/report.json')
                self.assertEqual(len(full['windows']),len(snapshots[0]['windows']))
                self.assertEqual(full['aggregate'],snapshots[0]['aggregate'])
                # Simulate an externally replaced complete snapshot, through
                # the same atomic writer that production uses.
                from cascade.preprocessing import atomic_json
                full['message']='refreshed snapshot'
                atomic_json(servers[0].demo_state.root/identity/'result/report.json',full)
                _,changed=request('GET','/api/jobs/'+identity)
                self.assertEqual(changed['report']['message'],'refreshed snapshot')
                self.assertEqual(changed['report']['aggregate'],full['aggregate'])

    def test_native_timeout_releases_process_and_replacement_can_respond(self):
        from cascade.streaming import Peer
        from cascade.tests.test_streaming import stalled_child
        peer=Peer('controlled',(),10,target=stalled_child)
        peer.timeout=.05
        try:
            with self.assertRaises(TimeoutError):peer.call('stall')
            self.assertFalse(peer.process.is_alive())
        finally:peer.close()
        replacement=Peer('controlled',(),10,target=responsive_child)
        try:
            self.assertEqual(replacement.call('echo',42),42)
            self.assertIn('scope',replacement.initial)
        finally:replacement.close()
        self.assertFalse(replacement.process.is_alive())


if __name__ == '__main__':
    unittest.main()

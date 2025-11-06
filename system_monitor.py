#!/usr/bin/env python3
"""
System Resource Monitoring System
실시간 시스템 리소스를 모니터링하고 시각화된 PDF 리포트를 생성합니다.
"""

import psutil
import time
import datetime
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')  # GUI 없이 실행
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import numpy as np
import os


class SystemMonitor:
    """시스템 리소스를 모니터링하는 클래스"""

    def __init__(self, duration_minutes=5, interval_seconds=2):
        """
        Args:
            duration_minutes: 모니터링 지속 시간 (분)
            interval_seconds: 데이터 수집 간격 (초)
        """
        self.duration = duration_minutes * 60
        self.interval = interval_seconds
        self.data = defaultdict(list)
        self.timestamps = []

    def get_cpu_info(self):
        """CPU 정보 수집"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        cpu_temps = self.get_cpu_temperature()

        return {
            'percent': cpu_percent,
            'frequency': cpu_freq.current if cpu_freq else 0,
            'temperature': cpu_temps
        }

    def get_cpu_temperature(self):
        """CPU 온도 수집 (가능한 경우)"""
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # 다양한 센서 이름 시도
                for name in ['coretemp', 'cpu_thermal', 'k10temp', 'zenpower']:
                    if name in temps:
                        return temps[name][0].current
                # 첫 번째 사용 가능한 온도 반환
                for sensor_name, entries in temps.items():
                    if entries:
                        return entries[0].current
        except (AttributeError, OSError):
            pass
        return None

    def get_memory_info(self):
        """메모리 정보 수집"""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return {
            'total': mem.total / (1024**3),  # GB
            'used': mem.used / (1024**3),
            'percent': mem.percent,
            'swap_total': swap.total / (1024**3),
            'swap_used': swap.used / (1024**3),
            'swap_percent': swap.percent
        }

    def get_disk_info(self):
        """디스크 정보 수집"""
        disk = psutil.disk_usage('/')
        io = psutil.disk_io_counters()

        return {
            'total': disk.total / (1024**3),  # GB
            'used': disk.used / (1024**3),
            'percent': disk.percent,
            'read_bytes': io.read_bytes / (1024**2) if io else 0,  # MB
            'write_bytes': io.write_bytes / (1024**2) if io else 0
        }

    def get_network_info(self):
        """네트워크 정보 수집"""
        net_io = psutil.net_io_counters()

        return {
            'bytes_sent': net_io.bytes_sent / (1024**2),  # MB
            'bytes_recv': net_io.bytes_recv / (1024**2),
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv
        }

    def get_gpu_info(self):
        """GPU 정보 수집 (가능한 경우)"""
        try:
            # nvidia-smi 사용 시도
            import subprocess
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                values = result.stdout.strip().split(',')
                return {
                    'temperature': float(values[0]),
                    'utilization': float(values[1]),
                    'memory_used': float(values[2]),
                    'memory_total': float(values[3])
                }
        except (FileNotFoundError, subprocess.TimeoutExpired, IndexError, ValueError):
            pass
        return None

    def collect_data(self):
        """데이터 수집 실행"""
        print(f"시스템 모니터링 시작 - {self.duration}초 동안 {self.interval}초 간격으로 수집")
        print("=" * 80)

        start_time = time.time()
        iteration = 0

        # 초기 네트워크 값 저장 (델타 계산용)
        prev_net = self.get_network_info()
        prev_disk_io = psutil.disk_io_counters()

        while time.time() - start_time < self.duration:
            iteration += 1
            current_time = datetime.datetime.now()
            self.timestamps.append(current_time)

            # CPU 정보
            cpu_info = self.get_cpu_info()
            self.data['cpu_percent'].append(cpu_info['percent'])
            self.data['cpu_freq'].append(cpu_info['frequency'])
            if cpu_info['temperature']:
                self.data['cpu_temp'].append(cpu_info['temperature'])

            # 메모리 정보
            mem_info = self.get_memory_info()
            self.data['mem_percent'].append(mem_info['percent'])
            self.data['mem_used'].append(mem_info['used'])
            self.data['swap_percent'].append(mem_info['swap_percent'])

            # 디스크 정보
            disk_info = self.get_disk_info()
            self.data['disk_percent'].append(disk_info['percent'])

            # 디스크 I/O (델타)
            curr_disk_io = psutil.disk_io_counters()
            if prev_disk_io:
                read_speed = (curr_disk_io.read_bytes - prev_disk_io.read_bytes) / (1024**2) / self.interval
                write_speed = (curr_disk_io.write_bytes - prev_disk_io.write_bytes) / (1024**2) / self.interval
                self.data['disk_read_speed'].append(max(0, read_speed))
                self.data['disk_write_speed'].append(max(0, write_speed))
            prev_disk_io = curr_disk_io

            # 네트워크 정보 (델타)
            curr_net = self.get_network_info()
            net_sent_speed = (curr_net['bytes_sent'] - prev_net['bytes_sent']) / self.interval
            net_recv_speed = (curr_net['bytes_recv'] - prev_net['bytes_recv']) / self.interval
            self.data['net_sent_speed'].append(max(0, net_sent_speed))
            self.data['net_recv_speed'].append(max(0, net_recv_speed))
            prev_net = curr_net

            # GPU 정보 (가능한 경우)
            gpu_info = self.get_gpu_info()
            if gpu_info:
                self.data['gpu_temp'].append(gpu_info['temperature'])
                self.data['gpu_util'].append(gpu_info['utilization'])
                self.data['gpu_mem'].append(gpu_info['memory_used'])

            # 진행 상황 출력
            elapsed = time.time() - start_time
            remaining = self.duration - elapsed
            print(f"[{iteration:3d}] {current_time.strftime('%H:%M:%S')} | "
                  f"CPU: {cpu_info['percent']:5.1f}% | "
                  f"MEM: {mem_info['percent']:5.1f}% | "
                  f"DISK: {disk_info['percent']:5.1f}% | "
                  f"남은 시간: {remaining:.0f}초")

            time.sleep(self.interval)

        print("=" * 80)
        print(f"데이터 수집 완료: {len(self.timestamps)}개 샘플")

    def generate_visualizations(self):
        """시각화 그래프 생성"""
        print("\n시각화 생성 중...")

        # 한글 폰트 설정 (폰트가 없으면 기본 폰트 사용)
        try:
            plt.rcParams['font.family'] = 'DejaVu Sans'
        except:
            pass

        plt.rcParams['axes.unicode_minus'] = False

        image_files = []

        # 1. CPU 사용률 및 주파수
        if self.data['cpu_percent']:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            ax1.plot(self.timestamps, self.data['cpu_percent'], 'b-', linewidth=2)
            ax1.fill_between(self.timestamps, self.data['cpu_percent'], alpha=0.3)
            ax1.set_ylabel('CPU Usage (%)', fontsize=12, fontweight='bold')
            ax1.set_title('CPU Usage Over Time', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 100])

            ax2.plot(self.timestamps, self.data['cpu_freq'], 'r-', linewidth=2)
            ax2.fill_between(self.timestamps, self.data['cpu_freq'], alpha=0.3, color='red')
            ax2.set_ylabel('Frequency (MHz)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()
            cpu_img = 'cpu_usage.png'
            plt.savefig(cpu_img, dpi=150, bbox_inches='tight')
            plt.close()
            image_files.append(cpu_img)

        # 2. CPU 온도
        if self.data['cpu_temp']:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(self.timestamps[:len(self.data['cpu_temp'])], self.data['cpu_temp'], 'orange', linewidth=2)
            ax.fill_between(self.timestamps[:len(self.data['cpu_temp'])], self.data['cpu_temp'], alpha=0.3, color='orange')
            ax.set_ylabel('Temperature (C)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12, fontweight='bold')
            ax.set_title('CPU Temperature', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            plt.tight_layout()
            temp_img = 'cpu_temperature.png'
            plt.savefig(temp_img, dpi=150, bbox_inches='tight')
            plt.close()
            image_files.append(temp_img)

        # 3. 메모리 사용률
        if self.data['mem_percent']:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(self.timestamps, self.data['mem_percent'], 'g-', linewidth=2, label='RAM Usage')
            ax.fill_between(self.timestamps, self.data['mem_percent'], alpha=0.3, color='green')
            if self.data['swap_percent'] and max(self.data['swap_percent']) > 0:
                ax.plot(self.timestamps, self.data['swap_percent'], 'purple', linewidth=2, label='Swap Usage')
            ax.set_ylabel('Usage (%)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12, fontweight='bold')
            ax.set_title('Memory Usage', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 100])
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            plt.tight_layout()
            mem_img = 'memory_usage.png'
            plt.savefig(mem_img, dpi=150, bbox_inches='tight')
            plt.close()
            image_files.append(mem_img)

        # 4. 디스크 I/O
        if self.data['disk_read_speed'] and self.data['disk_write_speed']:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(self.timestamps[:len(self.data['disk_read_speed'])],
                   self.data['disk_read_speed'], 'b-', linewidth=2, label='Read Speed')
            ax.plot(self.timestamps[:len(self.data['disk_write_speed'])],
                   self.data['disk_write_speed'], 'r-', linewidth=2, label='Write Speed')
            ax.fill_between(self.timestamps[:len(self.data['disk_read_speed'])],
                           self.data['disk_read_speed'], alpha=0.3, color='blue')
            ax.fill_between(self.timestamps[:len(self.data['disk_write_speed'])],
                           self.data['disk_write_speed'], alpha=0.3, color='red')
            ax.set_ylabel('Speed (MB/s)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12, fontweight='bold')
            ax.set_title('Disk I/O Speed', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            plt.tight_layout()
            disk_img = 'disk_io.png'
            plt.savefig(disk_img, dpi=150, bbox_inches='tight')
            plt.close()
            image_files.append(disk_img)

        # 5. 네트워크 트래픽
        if self.data['net_sent_speed'] and self.data['net_recv_speed']:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(self.timestamps[:len(self.data['net_sent_speed'])],
                   self.data['net_sent_speed'], 'purple', linewidth=2, label='Upload Speed')
            ax.plot(self.timestamps[:len(self.data['net_recv_speed'])],
                   self.data['net_recv_speed'], 'cyan', linewidth=2, label='Download Speed')
            ax.fill_between(self.timestamps[:len(self.data['net_sent_speed'])],
                           self.data['net_sent_speed'], alpha=0.3, color='purple')
            ax.fill_between(self.timestamps[:len(self.data['net_recv_speed'])],
                           self.data['net_recv_speed'], alpha=0.3, color='cyan')
            ax.set_ylabel('Speed (MB/s)', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time', fontsize=12, fontweight='bold')
            ax.set_title('Network Traffic', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            plt.tight_layout()
            net_img = 'network_traffic.png'
            plt.savefig(net_img, dpi=150, bbox_inches='tight')
            plt.close()
            image_files.append(net_img)

        # 6. GPU 정보 (가능한 경우)
        if self.data.get('gpu_temp') and self.data.get('gpu_util'):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

            ax1.plot(self.timestamps[:len(self.data['gpu_util'])],
                    self.data['gpu_util'], 'brown', linewidth=2)
            ax1.fill_between(self.timestamps[:len(self.data['gpu_util'])],
                            self.data['gpu_util'], alpha=0.3, color='brown')
            ax1.set_ylabel('GPU Usage (%)', fontsize=12, fontweight='bold')
            ax1.set_title('GPU Utilization', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([0, 100])

            ax2.plot(self.timestamps[:len(self.data['gpu_temp'])],
                    self.data['gpu_temp'], 'red', linewidth=2)
            ax2.fill_between(self.timestamps[:len(self.data['gpu_temp'])],
                            self.data['gpu_temp'], alpha=0.3, color='red')
            ax2.set_ylabel('Temperature (C)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
            ax2.set_title('GPU Temperature', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            for ax in [ax1, ax2]:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

            plt.tight_layout()
            gpu_img = 'gpu_info.png'
            plt.savefig(gpu_img, dpi=150, bbox_inches='tight')
            plt.close()
            image_files.append(gpu_img)

        print(f"시각화 완료: {len(image_files)}개 이미지 생성")
        return image_files

    def calculate_statistics(self):
        """통계 계산"""
        stats = {}

        if self.data['cpu_percent']:
            stats['CPU Usage'] = {
                'Average': f"{np.mean(self.data['cpu_percent']):.2f}%",
                'Min': f"{np.min(self.data['cpu_percent']):.2f}%",
                'Max': f"{np.max(self.data['cpu_percent']):.2f}%",
                'Std Dev': f"{np.std(self.data['cpu_percent']):.2f}%"
            }

        if self.data['cpu_temp']:
            stats['CPU Temperature'] = {
                'Average': f"{np.mean(self.data['cpu_temp']):.2f} C",
                'Min': f"{np.min(self.data['cpu_temp']):.2f} C",
                'Max': f"{np.max(self.data['cpu_temp']):.2f} C",
                'Std Dev': f"{np.std(self.data['cpu_temp']):.2f} C"
            }

        if self.data['mem_percent']:
            stats['Memory Usage'] = {
                'Average': f"{np.mean(self.data['mem_percent']):.2f}%",
                'Min': f"{np.min(self.data['mem_percent']):.2f}%",
                'Max': f"{np.max(self.data['mem_percent']):.2f}%",
                'Std Dev': f"{np.std(self.data['mem_percent']):.2f}%"
            }

        if self.data['disk_read_speed']:
            stats['Disk Read Speed'] = {
                'Average': f"{np.mean(self.data['disk_read_speed']):.2f} MB/s",
                'Min': f"{np.min(self.data['disk_read_speed']):.2f} MB/s",
                'Max': f"{np.max(self.data['disk_read_speed']):.2f} MB/s",
                'Total': f"{np.sum(self.data['disk_read_speed']) * self.interval:.2f} MB"
            }

        if self.data['disk_write_speed']:
            stats['Disk Write Speed'] = {
                'Average': f"{np.mean(self.data['disk_write_speed']):.2f} MB/s",
                'Min': f"{np.min(self.data['disk_write_speed']):.2f} MB/s",
                'Max': f"{np.max(self.data['disk_write_speed']):.2f} MB/s",
                'Total': f"{np.sum(self.data['disk_write_speed']) * self.interval:.2f} MB"
            }

        if self.data['net_sent_speed']:
            stats['Network Upload'] = {
                'Average': f"{np.mean(self.data['net_sent_speed']):.2f} MB/s",
                'Min': f"{np.min(self.data['net_sent_speed']):.2f} MB/s",
                'Max': f"{np.max(self.data['net_sent_speed']):.2f} MB/s",
                'Total': f"{np.sum(self.data['net_sent_speed']) * self.interval:.2f} MB"
            }

        if self.data['net_recv_speed']:
            stats['Network Download'] = {
                'Average': f"{np.mean(self.data['net_recv_speed']):.2f} MB/s",
                'Min': f"{np.min(self.data['net_recv_speed']):.2f} MB/s",
                'Max': f"{np.max(self.data['net_recv_speed']):.2f} MB/s",
                'Total': f"{np.sum(self.data['net_recv_speed']) * self.interval:.2f} MB"
            }

        if self.data.get('gpu_util'):
            stats['GPU Usage'] = {
                'Average': f"{np.mean(self.data['gpu_util']):.2f}%",
                'Min': f"{np.min(self.data['gpu_util']):.2f}%",
                'Max': f"{np.max(self.data['gpu_util']):.2f}%",
                'Std Dev': f"{np.std(self.data['gpu_util']):.2f}%"
            }

        if self.data.get('gpu_temp'):
            stats['GPU Temperature'] = {
                'Average': f"{np.mean(self.data['gpu_temp']):.2f} C",
                'Min': f"{np.min(self.data['gpu_temp']):.2f} C",
                'Max': f"{np.max(self.data['gpu_temp']):.2f} C",
                'Std Dev': f"{np.std(self.data['gpu_temp']):.2f} C"
            }

        return stats

    def generate_pdf_report(self, image_files, output_file='system_monitor_report.pdf'):
        """PDF 리포트 생성"""
        print(f"\nPDF 리포트 생성 중: {output_file}")

        doc = SimpleDocTemplate(output_file, pagesize=landscape(A4))
        story = []
        styles = getSampleStyleSheet()

        # 제목 스타일
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=1  # 중앙 정렬
        )

        # 부제목 스타일
        subtitle_style = ParagraphStyle(
            'CustomSubtitle',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2ca02c'),
            spaceAfter=12
        )

        # 제목
        title = Paragraph("System Resource Monitoring Report", title_style)
        story.append(title)

        # 모니터링 정보
        info_text = f"""
        <para align=center>
        <b>Monitoring Duration:</b> {self.duration / 60:.1f} minutes<br/>
        <b>Sample Interval:</b> {self.interval} seconds<br/>
        <b>Total Samples:</b> {len(self.timestamps)}<br/>
        <b>Start Time:</b> {self.timestamps[0].strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>End Time:</b> {self.timestamps[-1].strftime('%Y-%m-%d %H:%M:%S')}<br/>
        </para>
        """
        story.append(Paragraph(info_text, styles['Normal']))
        story.append(Spacer(1, 0.3*inch))

        # 시스템 정보
        story.append(Paragraph("System Information", subtitle_style))

        sys_info_data = [
            ['Category', 'Information'],
            ['CPU Cores', f'{psutil.cpu_count(logical=False)} Physical, {psutil.cpu_count(logical=True)} Logical'],
            ['Total Memory', f'{psutil.virtual_memory().total / (1024**3):.2f} GB'],
            ['Total Disk', f'{psutil.disk_usage("/").total / (1024**3):.2f} GB'],
            ['OS', f'{psutil.LINUX if hasattr(psutil, "LINUX") else "Unknown"}'],
        ]

        sys_table = Table(sys_info_data, colWidths=[2*inch, 5*inch])
        sys_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(sys_table)
        story.append(Spacer(1, 0.3*inch))

        # 통계 테이블
        story.append(Paragraph("Statistical Summary", subtitle_style))
        stats = self.calculate_statistics()

        stats_data = [['Metric', 'Average', 'Min', 'Max', 'Additional']]
        for metric, values in stats.items():
            row = [metric]
            row.append(values.get('Average', 'N/A'))
            row.append(values.get('Min', 'N/A'))
            row.append(values.get('Max', 'N/A'))
            row.append(values.get('Std Dev', values.get('Total', 'N/A')))
            stats_data.append(row)

        stats_table = Table(stats_data, colWidths=[2.2*inch, 1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.lightgrey, colors.white])
        ]))
        story.append(stats_table)
        story.append(PageBreak())

        # 그래프 추가
        story.append(Paragraph("Performance Graphs", subtitle_style))
        story.append(Spacer(1, 0.2*inch))

        for img_file in image_files:
            if os.path.exists(img_file):
                img = Image(img_file, width=9*inch, height=3*inch)
                story.append(img)
                story.append(Spacer(1, 0.2*inch))

        # PDF 생성
        doc.build(story)
        print(f"PDF 리포트 생성 완료: {output_file}")

        # 이미지 파일 정리
        for img_file in image_files:
            if os.path.exists(img_file):
                os.remove(img_file)


def main():
    """메인 실행 함수"""
    print("=" * 80)
    print("System Resource Monitoring System")
    print("=" * 80)
    print()

    # 모니터 생성 (1분간 1초 간격)
    monitor = SystemMonitor(duration_minutes=1, interval_seconds=1)

    # 데이터 수집
    monitor.collect_data()

    # 시각화 생성
    image_files = monitor.generate_visualizations()

    # PDF 리포트 생성
    monitor.generate_pdf_report(image_files)

    print("\n" + "=" * 80)
    print("모니터링 완료!")
    print("=" * 80)


if __name__ == "__main__":
    main()

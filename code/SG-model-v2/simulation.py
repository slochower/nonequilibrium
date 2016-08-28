#!/usr/bin/env python

from __future__ import division, print_function

import math as math
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.ndimage.filters import gaussian_filter
import scipy as sc

from aesthetics import *


class simulation(object):
    def plot_energy(self):
        """
        This function plots the unbound and bound energies associated with a simulation object.
        """
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(2, 2, wspace=0.2, hspace=0.5)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax1.plot(range(self.bins), self.unbound, c='r')
        ax1.set_title('Unbound chemical-potential-like energies', y=1.05)
        ax1.set_xlabel('Bins')
        ax2.plot(range(self.bins), self.bound, c='b')
        ax2.set_title('Bound chemical-potential-like energies', y=1.05)
        ax2.set_xlabel('Bins')
        fetching_plot(fig)
        plt.show()

    def plot_ss(self):
        """
        This function plots the steady-state distribution and Boltzmann PDF associated with a simulation object.
        By default, this will plot the eigenvector-derived steady-state distribution.
        """

        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(2, 2, wspace=0.4, hspace=0.5)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax1.plot(range(self.bins), self.ss[0:self.bins], c='r', label='SS')
        # ax1.plot(range(self.bins), self.PDF_unbound, c='k', label='PDF')
        # ax1.legend()
        ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
        ax1.set_title('Unbound steady-state', y=1.05)
        ax1.set_xlabel('Bins')

        ax2.plot(range(self.bins), self.ss[self.bins:2 * self.bins], c='b', label='SS')
        # ax2.plot(range(self.bins), self.PDF_bound, c='k', label='PDF')
        ax2.set_title('Bound steady-state', y=1.05)
        ax2.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
        # ax2.legend()
        ax2.set_xlabel('Bins')

        fetching_plot(fig)
        plt.show()

    def plot_flux(self, label=None):
        """
        This function plots the intrasurface flux sum and labels the graph with the attributions of the
        simulation object.
        """
        fig = plt.figure(figsize=(8, 6))
        gs = GridSpec(1, 1, wspace=0.2, hspace=0.5)
        ax1 = plt.subplot(gs[0, 0])
        ax1.plot(range(self.bins), self.flux_u, c='r', label='U')
        ax1.plot(range(self.bins), self.flux_b, c='b', label='B')
        ax1.plot(range(self.bins), self.flux_b + self.flux_u, c='k', label='U+B')
        ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
        # ax1.legend()
        ax1.set_xlabel('Bins')
        ax1.set_ylabel('Intrausrface flux (cycles/second)')
        ax1.set_title(r'{0:0.2f} $\pm$ {1:0.2f} cycles/second'.format(np.mean(self.flux_u + self.flux_b), sc.stats.sem(self.flux_u + self.flux_b)))
        if self.name is not None:
            ax1.set_title(self.name)
        print('{}: C_intrasurface = {:6.2e}, C_intersurface = {}, catalytic rate = {}, cATP = {}, dt = {}'.format(label,
                                                                                                                  self.C_intrasurface,
                                                                                                                  self.C_intersurface,
                                                                                                                  self.catalytic_rate,
                                                                                                                  self.cATP,
                                                                                                                  self.dt))
        fetching_plot(fig)
        plt.show()

    def plot_load(self):
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(2, 2, wspace=0.4, hspace=0.5)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])
        ax1.plot(range(self.bins), self.unbound, c='r')
        ax1.plot(range(self.bins), [self.unbound[i] + self.load_function(i) for i in range(self.bins)], c='k')
        ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
        ax1.set_title('Unbound', y=1.05)

        ax2.plot(range(self.bins), self.bound, c='b', label='SS')
        ax2.plot(range(self.bins), [self.bound[i] + self.load_function(i) for i in range(self.bins)], c='k',
                 label='PDF')
        ax2.set_title('Bound', y=1.05)
        ax2.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
        fetching_plot(fig)
        plt.show()

    def plot_load_extrapolation(self):
        fig = plt.figure(figsize=(12, 12))
        gs = GridSpec(2, 2, wspace=0.4, hspace=0.5)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])

        extended_u = np.tile(self.unbound, 4)
        extended_b = np.tile(self.bound, 4)

        ax1.plot(range(-2 * self.bins, 2 * self.bins),
                 [extended_u[i] + self.load_function(i) for i in np.arange(-2 * self.bins, 2 * self.bins)], c='k')
        print([extended_u[i] + self.load_function(i) for i in np.arange(-2 * self.bins, 2 * self.bins)])
        ax1.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
        ax1.set_title('Unbound', y=1.05)

        ax2.plot(range(-2 * self.bins, 2 * self.bins),
                 [extended_b[i] + self.load_function(i) for i in np.arange(-2 * self.bins, 2 * self.bins)], c='k')
        ax2.set_title('Bound', y=1.05)
        ax2.yaxis.set_major_formatter(mpl.ticker.ScalarFormatter(useMathText=True, useOffset=False))
        fetching_plot(fig)
        plt.show()

    def data_to_energy(self, histogram):
        """
        This function takes in population histograms from Chris' PKA data and
        (a) smoothes them with a Gaussian kernel with width 1;
        (b) eliminates zeros by setting any zero value to the minimum of the data;
        (c) turns the pouplation histograms to energy surfaces.
        """

        histogram_smooth = gaussian_filter(histogram, 1)
        histogram_copy = np.copy(histogram_smooth)
        for i in range(len(histogram)):
            if histogram_smooth[i] != 0:
                histogram_copy[i] = histogram_smooth[i]
            else:
                histogram_copy[i] = min(histogram_smooth[np.nonzero(histogram_smooth)])
        histogram_smooth = histogram_copy
        assert (not np.any(histogram_smooth == 0))
        histogram_smooth /= np.sum(histogram_smooth)
        energy = -self.kT * np.log(histogram_smooth)
        return energy

    def calculate_boltzmann(self):
        boltzmann_unbound = np.exp(-1 * np.array(self.unbound) / self.kT)
        boltzmann_bound = np.exp(-1 * np.array(self.bound) / self.kT)
        self.PDF_unbound = boltzmann_unbound / np.sum((boltzmann_unbound))
        self.PDF_bound = boltzmann_bound / np.sum((boltzmann_bound))

    def calculate_intrasurface_rates(self, energy_surface):
        """
        This function calculates intrasurface rates using the energy difference between adjacent bins.
        """
        forward_rates = self.C_intrasurface * np.exp(-1 * np.diff(energy_surface) / float(2 * self.kT))
        backward_rates = self.C_intrasurface * np.exp(+1 * np.diff(energy_surface) / float(2 * self.kT))
        rate_matrix = np.zeros((self.bins, self.bins))
        for i in range(self.bins - 1):
            rate_matrix[i][i + 1] = forward_rates[i]
            rate_matrix[i + 1][i] = backward_rates[i]
        rate_matrix[0][self.bins - 1] = self.C_intrasurface * np.exp(
            -(energy_surface[self.bins - 1] - energy_surface[0]) / float(2 * self.kT))
        rate_matrix[self.bins - 1][0] = self.C_intrasurface * np.exp(
            +(energy_surface[self.bins - 1] - energy_surface[0]) / float(2 * self.kT))
        return rate_matrix

    def calculate_intrasurface_rates_with_load(self, energy_surface):
        """
        This function calculates intrasurface rates using the energy difference between adjacent bins.
        """

        surface_with_load = [energy_surface[i] + self.load_function(i) for i in range(self.bins)]
        # This should handle the interior elements just fine.
        self.forward_rates = self.C_intrasurface * \
                             np.exp(-1 * np.diff(surface_with_load) / float(2 * self.kT))
        self.backward_rates = self.C_intrasurface * \
                              np.exp(+1 * np.diff(surface_with_load) / float(2 * self.kT))
        rate_matrix = np.zeros((self.bins, self.bins))
        for i in range(self.bins - 1):
            rate_matrix[i][i + 1] = self.forward_rates[i]
            rate_matrix[i + 1][i] = self.backward_rates[i]

        # But now the PBCs are a little tricky...
        rate_matrix[0][self.bins - 1] = self.C_intrasurface * np.exp(
            -(energy_surface[self.bins - 1] + self.load_function(-1) -
              (energy_surface[0] + self.load_function(0))) / float(2 * self.kT))

        rate_matrix[self.bins - 1][0] = self.C_intrasurface * np.exp(
            +(energy_surface[self.bins - 1] + self.load_function(self.bins - 1) -
              (energy_surface[0] + self.load_function(self.bins))) / float(2 * self.kT))

        return rate_matrix

    def calculate_intersurface_rates(self, unbound_surface, bound_surface):
        """
        This function calculates the intersurface rates in two ways.
        For bound to unbound, the rates are calculated according to the energy difference and the catalytic rate.
        For unbound to bound, the rates depend on the prefactor and the concentration of ATP.
        """
        bu_rm = np.empty((self.bins))
        ub_rm = np.empty((self.bins))
        for i in range(self.bins):
            bu_rm[i] = (self.C_intersurface *
                        np.exp(-1 * (unbound_surface[i] - bound_surface[i]) / float(self.kT)) +
                        self.catalytic_rate)
            ub_rm[i] = self.C_intersurface * self.cATP
        return ub_rm, bu_rm

    def compose_tm(self, u_rm, b_rm, ub_rm, bu_rm):
        """
        We take the four rate matrices (two single surface and two intersurface) and inject them into the transition matrix.
        """
        if self.extra_precision:
            tm = np.zeros((2 * self.bins, 2 * self.bins), dtype=np.longdouble)
        else:
            tm = np.zeros((2 * self.bins, 2 * self.bins))
        tm[0:self.bins, 0:self.bins] = u_rm
        tm[self.bins:2 * self.bins, self.bins:2 * self.bins] = b_rm
        for i in range(self.bins):
            tm[i, i + self.bins] = ub_rm[i]
            tm[i + self.bins, i] = bu_rm[i]
        self.tm = self.scale_tm(tm)
        return

    def scale_tm(self, tm):
        """
        The transition matrix is scaled by `dt` so all rows sum to 1 and all elements are less than 1.
        This should not use `self` subobjects, except for `dt` because we are mutating the variables.
        """
        row_sums = tm.sum(axis=1, keepdims=True)
        maximum_row_sum = int(math.log10(max(row_sums)))
        self.dt = 10 ** -(maximum_row_sum + 1)
        tm_scaled = self.dt * tm
        row_sums = tm_scaled.sum(axis=1, keepdims=True)
        if np.any(row_sums > 1):
            print('Row sums unexpectedly greater than 1.')
        for i in range(2 * self.bins):
            tm_scaled[i][i] = 1.0 - row_sums[i]
        return tm_scaled

    def calculate_eigenvector(self):
        """
        The eigenvectors and eigenvalues of the transition matrix are computed and the steady-state population is
        assigned to the eigenvector with an eigenvalue of 1.
        """
        self.eigenvalues, eigenvectors = np.linalg.eig(np.transpose(self.tm))
        ss = abs(eigenvectors[:, self.eigenvalues.argmax()].astype(float))
        self.ss = ss / np.sum(ss)
        return

    def calculate_flux(self, ss, tm):
        """
        This function calculates the intrasurface flux using the steady-state distribution and the transition matrix.
        The steady-state distribution is a parameter so this function can be run with either the eigenvector-derived
        steady-state distribution or the interated steady-state distribution.
        """
        flux_u = np.empty((self.bins))
        flux_b = np.empty((self.bins))
        for i in range(self.bins):
            if i == 0:
                flux_u[i] = -1 * (- ss[i] * tm[i][i + 1] / self.dt + ss[i + 1] * tm[i + 1][i] / self.dt)
            if i == self.bins - 1:
                flux_u[i] = -1 * (- ss[i] * tm[i][0] / self.dt + ss[0] * tm[0][i] / self.dt)
            else:
                flux_u[i] = -1 * (- ss[i] * tm[i][i + 1] / self.dt + ss[i + 1] * tm[i + 1][i] / self.dt)
        for i in range(self.bins, 2 * self.bins):
            if i == self.bins:
                flux_b[i - self.bins] = -1 * (- ss[i] * tm[i][i + 1] / self.dt + ss[i + 1] * tm[i + 1][i] / self.dt)
            if i == 2 * self.bins - 1:
                flux_b[i - self.bins] = -1 * (
                    - ss[i] * tm[i][self.bins] / self.dt + ss[self.bins] * tm[self.bins][i] / self.dt)
            else:
                flux_b[i - self.bins] = -1 * (- ss[i] * tm[i][i + 1] / self.dt + ss[i + 1] * tm[i + 1][i] / self.dt)
        self.flux_u = flux_u
        self.flux_b = flux_b
        return

    def iterate(self, iterations=None):
        """
        A template population distribution is multiplied by the transition matrix.
        By default, iterations are 0. The output of this command is set to
        `self.iterative_ss` which can be passed to `calculate_flux`.
        The new population can be set to a normalized random distribution or the eigenvector-derived
        steady-state distribution.
        """
        print('Running iterative method with {} iterations'.format(self.iterations))
        population = np.random.rand(2 * self.bins)
        row_sums = population.sum(axis=0, keepdims=True)
        population = population / row_sums
        iterations = 0
        if self.ss is not None:
            new_population = self.ss
        for i in range(self.iterations):
            new_population = np.dot(new_population, self.tm)
        self.iterative_ss = new_population
        return

    def simulate(self, plot=False, debug=False, pka_md_data=True):
        """
        Now this function takes in a file(name) and determins the energy surfaces automatically,
        so I don't forget to do it in an interactive session.
        This function runs the `simulation` which involves:
        (a) setting the unbound intrasurface rates,
        (b) setting the bound intrasurface rates,
        (c) setting the intersurface rates,
        (d) composing the transition matrix,
        (e) calculating the eigenvectors of the transition matrix,
        (f) calculating the intrasurface flux,
        and optionally (g) running an interative method to determine the steady-state distribution.
        """
        if pka_md_data:
            try:
                self.unbound_population = np.genfromtxt(self.dir + '/apo/' + self.name + '_chi_pop_hist_targ.txt',
                                                        delimiter=',',
                                                        skip_header=1)
                self.bound_population = np.genfromtxt(self.dir + '/atpmg/' + self.name + '_chi_pop_hist_ref.txt',
                                                      delimiter=',',
                                                      skip_header=1)
            except IOError:
                print('Cannot read {} from {}.'.format(self.name, self.dir))
            except:
                print('Unknown error.')
        else:
            # Populations are supplied manually.
            pass

        self.unbound = self.data_to_energy(self.unbound_population)
        self.bound = self.data_to_energy(self.bound_population) - self.offset_factor

        self.bins = len(self.unbound)
        self.tm = np.zeros((self.bins, self.bins))
        self.C_intrasurface = self.C_intrasurface_0 / (360. / self.bins) ** 2  # per degree per second

        if not self.load:
            u_rm = self.calculate_intrasurface_rates(self.unbound)
            b_rm = self.calculate_intrasurface_rates(self.bound)
        if self.load:
            u_rm = self.calculate_intrasurface_rates_with_load(self.unbound)
            b_rm = self.calculate_intrasurface_rates_with_load(self.bound)

        ub_rm, bu_rm = self.calculate_intersurface_rates(self.unbound, self.bound)
        self.compose_tm(u_rm, b_rm, ub_rm, bu_rm)
        self.calculate_eigenvector()
        self.calculate_boltzmann()
        self.calculate_flux(self.ss, self.tm)
        if plot:
            if not self.load:
                self.plot_energy()
            else:
                self.plot_load()
            self.plot_ss()
            self.plot_flux(label='Eigenvector method')
        if self.iterations != 0:
            self.iterate(self.iterations)
            self.calculate_flux(self.iterative_ss, self.tm)
            if plot:
                self.plot_flux(label='Iterative method')
        if debug:
            self.parameters = {
                'C_intersurface': self.C_intersurface,
                'C_intrasurface': self.C_intrasurface,
                'kT': self.kT,
                'cATP': self.cATP,
                'offset_factor': self.offset_factor,
                'catalytic rate': self.catalytic_rate,
                'iterations': self.iterations,
                'load': self.load,
                'load_slope': self.load_slope,
                'bins': self.bins,
                'steady_state': self.ss,
                'flux': self.flux_u + self.flux_b,
                'unbound_energy': self.unbound,
                'bound_energy': self.bound,
                'transition_matrix': self.tm,
                'dt': self.dt,
                'eigenvalues': self.eigenvalues,
                'PDF_unbound': self.PDF_unbound,
                'PDF_bound': self.PDF_bound,
                'extra_precision': self.extra_precision
            }
        return

    def load_function(self, x):
        return x * self.load_slope / self.bins

    def __init__(self):
        """
        These values are assigned to a new object, unless overridden later.
        """
        self.dir = 'pka-md-data'
        self.name = None
        self.kT = 0.6  # RT = 0.6 kcal per mol
        self.C_intrasurface_0 = 1.71 * 10 ** 14  # degree per second
        self.C_intersurface = 0.24 * 10 ** 6  # per mole per second
        self.cATP = 2 * 10 ** -3  # molar
        self.offset_factor = 6.25  # kcal per mol
        self.catalytic_rate = 140  # per second
        self.iterations = 0
        self.extra_precision = False
        self.load = False
        if self.load:
            self.load_slope = self.bins  # kcal per mol per (2 * pi) radians
        else:
            self.load_slope = 0

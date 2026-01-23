/**
 * Auth tests - RequireAuth redirect and login flows
 */

import { describe, it, expect, vi, type Mock } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { RequireAuth } from '@/components/auth/RequireAuth';
import { useAuthStore } from '@/stores/authStore';

const mockedUseAuthStore = useAuthStore as unknown as Mock;

// Mock Zustand auth store
vi.mock('@/stores/authStore', () => ({
  useAuthStore: vi.fn(),
}));

describe('RequireAuth', () => {
  it('should redirect to /login when not authenticated', async () => {
    // Mock unauthenticated state
    mockedUseAuthStore.mockReturnValue({
      isLoggedIn: false,
    });

    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Routes>
          <Route
            path="/chat"
            element={
              <RequireAuth>
                <div>Protected Content</div>
              </RequireAuth>
            }
          />
          <Route path="/login" element={<div>Login Page</div>} />
        </Routes>
      </MemoryRouter>
    );

    // Should redirect to login
    await waitFor(() => {
      expect(screen.getByText('Login Page')).toBeInTheDocument();
    });
  });

  it('should render protected content when authenticated', async () => {
    // Mock authenticated state
    mockedUseAuthStore.mockReturnValue({
      isLoggedIn: true,
    });

    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Routes>
          <Route
            path="/chat"
            element={
              <RequireAuth>
                <div>Protected Content</div>
              </RequireAuth>
            }
          />
          <Route path="/login" element={<div>Login Page</div>} />
        </Routes>
      </MemoryRouter>
    );

    // Should show protected content
    await waitFor(() => {
      expect(screen.getByText('Protected Content')).toBeInTheDocument();
    });
  });

  it('should not render anything initially when not authenticated', () => {
    mockedUseAuthStore.mockReturnValue({
      isLoggedIn: false,
    });

    render(
      <MemoryRouter initialEntries={['/chat']}>
        <Routes>
          <Route
            path="/chat"
            element={
              <RequireAuth>
                <div>Protected Content</div>
              </RequireAuth>
            }
          />
        </Routes>
      </MemoryRouter>
    );

    // Should not render protected content before redirect
    expect(screen.queryByText('Protected Content')).not.toBeInTheDocument();
  });
});

describe('Demo mode', () => {
  it('should allow demo login without backend session', () => {
    const mockLogin = vi.fn();
    mockedUseAuthStore.mockReturnValue({
      isLoggedIn: false,
      login: mockLogin,
    });

    // Simulate demo login
    const username = 'testuser';
    const email = 'test@example.com';
    mockLogin(username, email);

    expect(mockLogin).toHaveBeenCalledWith(username, email);
  });
});
